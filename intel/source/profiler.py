import os
import time

try:
    import pynvml
except Exception:  # no NVIDIA driver bindings (e.g. Intel GPUs)
    pynvml = None

from .energy_intel import (xpu_available, xpu_device_count, get_level_zero,
                           IntelEnergyMonitor, IntelPowerPoller)

try:
    from zeus.monitor import ZeusMonitor
except Exception:  # zeus not installed — fall back to the NVML-only estimate
    ZeusMonitor = None

try:
    from zeus.monitor.power import PowerMonitor, PowerDomain
except Exception:  # older zeus, or zeus missing entirely
    PowerMonitor = None
    PowerDomain = None


# A ZeusMonitor initialises NVML and, on GPUs without an energy counter, spawns
# a polling subprocess. model.py builds a Profiler per generate() call, so the
# monitor is cached here instead of being rebuilt for every request.
_MONITORS = {}
_WINDOW_SEQ = 0
_ANNOUNCED = set()

# One PowerMonitor covers every visible GPU for the whole process; each Profiler
# sums only over the devices its engine occupies. Building one per (tp, pp)
# device set would spawn a polling subprocess per config for no benefit.
_POWER_MONITOR = None
_POWER_MONITOR_READY = False
# Retained samples per GPU. Bounded so the per-window scan stays O(1) over a
# multi-hour run; dedup means a steady GPU contributes almost nothing, and 8192
# changes is far more history than one generate() window can span.
_POWER_SAMPLE_CAP = 8192


def _get_power_monitor():
    """Lazily start the shared power poller. ``None`` if unavailable."""
    global _POWER_MONITOR, _POWER_MONITOR_READY
    if _POWER_MONITOR_READY:
        return _POWER_MONITOR
    _POWER_MONITOR_READY = True
    if PowerMonitor is None:
        return None
    try:
        _POWER_MONITOR = PowerMonitor(
            gpu_indices=None,
            # DEVICE_INSTANT only: DEVICE_AVERAGE is the driver's own rolling
            # mean over an unspecified window, which would smear across window
            # boundaries, and each extra domain costs another subprocess.
            power_domains=[PowerDomain.DEVICE_INSTANT],
            max_samples_per_gpu=_POWER_SAMPLE_CAP,
        )
    except Exception as e:
        print(f"[Profiler] Zeus PowerMonitor unavailable ({e}); "
              f"falling back to two-point NVML power on device 0.")
        _POWER_MONITOR = None
    return _POWER_MONITOR


def _step_mean(samples, t0, t1):
    """Time-weighted mean of a deduplicated power timeline over [t0, t1].

    Zeus's poller only emits a sample when the wattage *changes*, so the
    timeline is a step function sampled at its own discontinuities, not a
    uniform series. Averaging the values arithmetically would weight a 20 ms
    spike the same as a 5 s plateau; this holds each reading until the next one
    (zero-order hold) and integrates.

    ``samples`` must be sorted by timestamp. Returns None if nothing is known
    about the window.
    """
    held = None          # value in force at t0: the last change at or before it
    inside = []
    for ts, w in samples:
        if ts <= t0:
            held = w
        elif ts <= t1:
            inside.append((ts, w))
    if held is None:
        if not inside:
            return None
        # Window opened before the first retained sample; back-fill with it.
        held = inside[0][1]

    span = t1 - t0
    if span <= 0:
        return held

    total, cur_t, cur_w = 0.0, t0, held
    for ts, w in inside:
        total += cur_w * (ts - cur_t)
        cur_t, cur_w = ts, w
    total += cur_w * (t1 - cur_t)
    return total / span


# Intel GPUs: Zeus has no Intel backend, so energy comes from the Level
# Zero Sysman energy counters instead (source/energy_intel.py), through the same
# begin_window/end_window interface. Decided once per process.
_INTEL = None
_INTEL_POLLER = None


def _intel():
    global _INTEL
    if _INTEL is None:
        _INTEL = get_level_zero(xpu_device_count()) if xpu_available() else False
    return _INTEL or None


def _intel_poller():
    global _INTEL_POLLER
    if _INTEL_POLLER is None and _intel() is not None:
        _INTEL_POLLER = IntelPowerPoller(_intel())
    return _INTEL_POLLER


def _get_monitor(gpu_indices, sync_with):
    ze = _intel()
    if ze is not None:
        key = ("intel", tuple(gpu_indices) if gpu_indices is not None else None)
        if key not in _MONITORS:
            idx = list(gpu_indices) if gpu_indices is not None else list(range(len(ze.visible)))
            try:
                _MONITORS[key] = IntelEnergyMonitor(ze, idx)
                _intel_poller()   # start polling now, so the first window has power samples
            except Exception as e:
                print(f"[Profiler] Level Zero monitor unavailable for devices {idx} ({e})")
                _MONITORS[key] = None
        return _MONITORS[key]
    if ZeusMonitor is None:
        return None
    key = (tuple(gpu_indices) if gpu_indices is not None else None, sync_with)
    if key not in _MONITORS:
        try:
            mon = ZeusMonitor(
                gpu_indices=list(gpu_indices) if gpu_indices is not None else None,
                sync_execution_with=sync_with,
            )
            # A monitor over zero devices reports 0 J for every window without
            # raising, which would silently fill a whole run with zeros.
            if not getattr(mon, "gpu_indices", None):
                raise RuntimeError("monitor covers no GPUs")
            _MONITORS[key] = mon
        except Exception as e:
            print(f"[Profiler] Zeus unavailable ({e}); "
                  f"falling back to NVML power sampling on device 0.")
            _MONITORS[key] = None
    return _MONITORS[key]


class Profiler:
    """Profiler that measures GPU energy and latency around a callable.

    Energy is measured with Zeus (https://github.com/ml-energy/zeus), which
    reads NVML's hardware energy accumulator
    (``nvmlDeviceGetTotalEnergyConsumption``) at the window boundaries. On
    Volta and newer the GPU integrates that counter internally at high
    frequency, so the value is a true measurement over the whole window and
    covers every GPU taking part in the run. Zeus falls back to a polling
    subprocess on older cards.

    This replaces an earlier scheme that sampled instantaneous power on device
    0 only, once before and once after the call, and reported
    ``(p_before + p_after) / 2 * latency``. That underestimated every TP/PP > 1
    run by roughly the GPU count, and the two-point average was biased because
    ``p_before`` is read while the GPU is still idle.

    ``avg_power_watts`` is the time-weighted mean of Zeus's polled instantaneous
    power over the same window, summed over the same GPUs. It stays a *sampled*
    quantity, read from NVML's power sensor by a background poller and never
    derived from the energy counter: computing ``energy / latency`` would make
    the MaverIQ baseline (latency x power) reproduce the energy column by
    construction and invalidate the cross-model comparison.

    It previously sampled device 0 once before and once after the call and
    reported ``(p_before + p_after) / 2``. That is retained only as a fallback,
    and it is wrong in two ways the poller fixes: on TP/PP > 1 it saw one GPU of
    several, so ``latency x power`` was short by roughly the GPU count and could
    not be compared against an energy column that covers all of them; and
    ``p_before`` is read while the GPU is still idle, biasing the mean low.

    Metrics collected:
    - total_latency (s)
    - avg_power_watts (W)          time-weighted, summed over the measured GPUs
    - power_source (str)           'zeus_sampling' or 'nvml_two_point'
    - n_power_samples (int)        readings backing avg_power_watts
    - energy_joules (J)            Zeus, summed over every measured GPU
    - energy_joules_per_gpu (dict) per-device breakdown
    - n_gpus_measured (int)
    - energy_source (str)          'nvml_counter' or 'nvml_power_sampling'
    - memory_used_before (bytes)
    - memory_used_after (bytes)
    - gpu_util_before (percent)
    - gpu_util_after (percent)

    Args:
        handle: optional NVML handle for the sampled power/memory/util fields.
        gpu_indices: devices to measure for energy. ``None`` measures every
            visible GPU, which is what the scheduler's CUDA_VISIBLE_DEVICES
            already restricts to the job's allocation.
        sync_with: framework used to synchronise at the window boundaries.
    """

    def __init__(self, handle=None, gpu_indices=None, sync_with="torch"):
        self.handle = handle
        self.gpu_indices = gpu_indices
        self.monitor = _get_monitor(gpu_indices, sync_with)
        self._announce()

    def _announce(self):
        """Log the measurement setup once per GPU set, for run provenance.

        Each (tp, pp) config builds a new engine over a different device set,
        so this fires once per config rather than once per process.
        """
        key = tuple(self.gpu_indices) if self.gpu_indices is not None else None
        if key in _ANNOUNCED:
            return
        _ANNOUNCED.add(key)
        if self.monitor is None:
            print("[Profiler] energy source: NVML power sampling (device 0 only)")
            return
        if isinstance(self.monitor, IntelEnergyMonitor):
            print(f"[Profiler] energy source: Level Zero energy counters, devices "
                  f"{self.monitor.gpu_indices} -> domains "
                  f"{[d.name for d in self.monitor.domains]} (scope {self.monitor.scope})\n"
                  f"[Profiler] power source:  counter differences polled every "
                  f"{_intel_poller().period_s}s, same domains")
            return
        try:
            idx = sorted(self.monitor.gpu_indices)
        except Exception:
            idx = "?"
        mon = _get_power_monitor()
        power = ("Zeus polled instant power"
                 f" @ {getattr(mon, 'update_period', '?')}s" if mon is not None
                 else "two-point NVML sampling (device 0)")
        print(f"[Profiler] energy source: Zeus / NVML energy counter, "
              f"GPUs {idx}  (CUDA_VISIBLE_DEVICES="
              f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})\n"
              f"[Profiler] power source:  {power}, summed over GPUs {idx}")

    def _ensure_handle(self):
        if pynvml is None or _intel() is not None:
            return
        if self.handle is None:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.handle = None

    def _safe_power(self):
        if pynvml is None:
            return 0.0
        try:
            return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except Exception:
            return 0.0

    def _window_power(self, t0, t1):
        """Mean power over [t0, t1], summed across this engine's GPUs.

        Returns (watts, n_samples), or (None, 0) when the poller has nothing to
        say about the window — the caller then falls back to two-point sampling.
        """
        if isinstance(self.monitor, IntelEnergyMonitor):
            poller = _intel_poller()
            total, n = 0.0, 0
            for d in self.monitor.domains:
                samples = poller.timeline(d.name)
                mean = _step_mean(samples, t0, t1)
                if mean is not None:
                    total += mean
                    n += len(samples)
            return (total, n) if n else (None, 0)
        mon = _get_power_monitor()
        if mon is None:
            return None, 0
        try:
            # No start_time filter: the reading in force at t0 may predate the
            # window by a long time when power is steady. The deque cap bounds
            # the scan instead.
            timelines = mon.get_power_timeline(
                PowerDomain.DEVICE_INSTANT, end_time=t1)
        except Exception as e:
            print(f"[Profiler] power timeline unavailable ({e}); "
                  f"using two-point sampling for this call.")
            return None, 0

        indices = (self.gpu_indices if self.gpu_indices is not None
                   else list(timelines))
        total, n = 0.0, 0
        for i in indices:
            samples = timelines.get(i) or []
            mean = _step_mean(samples, t0, t1)
            if mean is not None:
                total += mean
                n += len(samples)
        return (total, n) if n else (None, 0)

    def _safe_memory(self):
        if pynvml is None:
            return 0
        try:
            mi = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return mi.used
        except Exception:
            return 0

    def _safe_util(self):
        if pynvml is None:
            return 0
        try:
            ur = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return ur.gpu
        except Exception:
            return 0

    def profile(self, func, *args, **kwargs):
        """Run `func(*args, **kwargs)` while collecting energy and NVML metrics.

        Returns (result, metrics_dict).
        """
        global _WINDOW_SEQ

        # ensure NVML handle if possible
        self._ensure_handle()

        power_before = self._safe_power()
        mem_before = self._safe_memory()
        util_before = self._safe_util()

        _WINDOW_SEQ += 1
        window = f"profile_{_WINDOW_SEQ}"
        measurement = None

        if self.monitor is not None:
            try:
                self.monitor.begin_window(window)
            except Exception as e:
                print(f"[Profiler] begin_window failed ({e}); "
                      f"reverting to power sampling.")
                self.monitor = None

        t0 = time.time()
        try:
            result = func(*args, **kwargs)
        except Exception:
            # Never leave a window open: the driver catches generation errors
            # and keeps going, and a stale window would corrupt the next one.
            if self.monitor is not None:
                try:
                    self.monitor.end_window(window, cancel=True)
                except Exception:
                    pass
            raise
        t1 = time.time()

        if self.monitor is not None:
            try:
                measurement = self.monitor.end_window(window)
            except Exception as e:
                print(f"[Profiler] end_window failed ({e}); "
                      f"using power sampling for this call.")
                measurement = None

        power_after = self._safe_power()
        mem_after = self._safe_memory()
        util_after = self._safe_util()

        latency = t1 - t0

        intel = isinstance(self.monitor, IntelEnergyMonitor)
        sampled_power, n_power = self._window_power(t0, t1)
        if sampled_power is not None:
            avg_power = sampled_power
            power_source = "ze_counter_poll" if intel else "zeus_sampling"
        else:
            # Device 0 only, and biased low because power_before is read while
            # the GPU is idle. Kept so a run never loses the column outright.
            avg_power = (power_before + power_after) / 2.0
            power_source = "nvml_two_point"

        if measurement is not None:
            per_gpu = {int(k): float(v) for k, v in measurement.gpu_energy.items()}
            energy = float(measurement.total_energy)
            # Zeus times the same window; prefer it over the Python-side clock.
            latency = float(measurement.time) or latency
            source = f"ze_counter_{self.monitor.scope}" if intel else "nvml_counter"
        else:
            # avg_power already covers every GPU the engine uses, so this stays
            # on the same footing as the counter path it stands in for.
            per_gpu = {0: avg_power * latency}
            energy = avg_power * latency
            source = "nvml_power_sampling"

        metrics = {
            "total_latency": latency,
            "avg_power_watts": avg_power,
            "power_source": power_source,
            "n_power_samples": n_power,
            "energy_joules": energy,
            "energy_joules_per_gpu": per_gpu,
            "n_gpus_measured": len(per_gpu),
            "energy_source": source,
            "memory_used_before": mem_before,
            "memory_used_after": mem_after,
            "gpu_util_before": util_before,
            "gpu_util_after": util_after,
        }

        return result, metrics
