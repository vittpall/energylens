"""GPU energy on AMD Instinct (MI100) from the amdsmi hardware energy accumulator.

Zeus (used on NVIDIA, source/profiler.py) ships an AMD backend, but it needs
``amdsmi_get_gpu_enumeration_info`` (newer amdsmi) and, for the energy of a
window, falls back to a polling subprocess that integrates power. Neither is
acceptable here, so the same method as on NVIDIA is reproduced directly:
``amdsmi_get_energy_count`` returns the accumulator and its resolution
(uJ per count); the energy of a window is the difference between two reads,
summed over every GPU the engine occupies.

Power is a separate quantity: a background thread polls the firmware's socket
power sensor (``amdsmi_get_power_info``, W) every 100 ms. It is never derived
from the energy counter, so energy ~ power x latency stays a real check
(power_source = "amdsmi_sampling"). MI100 reports ``average_socket_power`` (a
firmware rolling mean) and ``current_socket_power`` as N/A; the former is used.

``HIP_VISIBLE_DEVICES`` / ``ROCR_VISIBLE_DEVICES`` are honoured so that engine
device i maps to the i-th visible physical GPU.
"""
import os
import threading
import time

try:
    import amdsmi
except Exception:
    amdsmi = None


# amdsmi_get_energy_count reports counter_resolution = 15.3 "uJ" per count on MI100,
# but a controlled load shows the accumulator advances by ~1 J per count: a 5 s
# GEMM load on one GPU at 279 W (sampled) moved the raw count by 1413 while the
# idle sibling moved by ~161 (32 W), i.e. 15.3 uJ x 2**16 = 1.003 J per count
# (test_amd_energy.py reproduces this). The raw value is a 16.16 fixed-point count.
# ENERGYLENS_AMD_COUNTER_SCALE overrides (1 gives the unscaled vendor value).
COUNTER_SCALE = float(os.environ.get("ENERGYLENS_AMD_COUNTER_SCALE", 2 ** 16))


def _visible_physical(n_total):
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        v = os.environ.get(var, "").strip()
        if v:
            try:
                return [int(x) for x in v.split(",") if x.strip() != ""]
            except ValueError:
                break
    return list(range(n_total))


class AMDSmi:
    """Process-wide amdsmi session: one handle per physical GPU."""

    def __init__(self):
        if amdsmi is None:
            raise RuntimeError("amdsmi python module not available")
        amdsmi.amdsmi_init()
        self.handles = list(amdsmi.amdsmi_get_processor_handles())
        if not self.handles:
            raise RuntimeError("amdsmi reports no GPUs")
        self.visible = _visible_physical(len(self.handles))

    def handle(self, device_index):
        return self.handles[self.visible[device_index]]

    def read_energy(self, device_index):
        """(energy J, timestamp s) of the hardware accumulator."""
        r = amdsmi.amdsmi_get_energy_count(self.handle(device_index))
        return float(r["power"]) * float(r["counter_resolution"]) * 1e-6 * COUNTER_SCALE, r["timestamp"] * 1e-9

    def read_power(self, device_index):
        """Socket power in W from the firmware sensor, or None."""
        info = amdsmi.amdsmi_get_power_info(self.handle(device_index))
        for key in ("current_socket_power", "average_socket_power"):
            v = info.get(key)
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
        return None


class _Measurement:
    def __init__(self, t, per_device):
        self.time = t
        self.gpu_energy = per_device
        self.total_energy = sum(per_device.values())


class AMDEnergyMonitor:
    """Zeus-like window API: begin_window / end_window over a set of GPUs."""

    def __init__(self, smi, device_indices):
        self.smi = smi
        self.gpu_indices = list(device_indices)
        self.scope = "gpu"
        self._open = {}

    def _read_all(self):
        # Median of three back-to-back reads: about 2% of the first Mistral-7B windows
        # had a boundary read that was stale (0 J over a 4.7 s window at 238 W) or wild
        # (100-800x power x latency). Three reads take < 1 ms and reject a single bad one.
        out = {}
        for i in self.gpu_indices:
            reads = sorted(self.smi.read_energy(i)[0] for _ in range(3))
            out[i] = reads[1]
        return out

    def begin_window(self, key):
        self._open[key] = (time.time(), self._read_all())

    def end_window(self, key, cancel=False):
        t0, start = self._open.pop(key)
        if cancel:
            return None
        end = self._read_all()
        t1 = time.time()
        return _Measurement(t1 - t0, {i: max(end[i] - start[i], 0.0) for i in self.gpu_indices})


class AMDPowerPoller:
    """Background thread: socket power per GPU, sampled every period_s."""

    def __init__(self, smi, period_s=0.1, cap=8192):
        self.smi, self.period_s, self.cap = smi, period_s, cap
        self.update_period = period_s
        self.timelines = {i: [] for i in range(len(smi.visible))}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            now = time.time()
            for i in self.timelines:
                try:
                    w = self.smi.read_power(i)
                except Exception:
                    continue
                if w is None:
                    continue
                with self._lock:
                    tl = self.timelines[i]
                    tl.append((now, w))
                    if len(tl) > self.cap:
                        del tl[: len(tl) - self.cap]
            self._stop.wait(self.period_s)

    def timeline(self, device_index):
        with self._lock:
            return list(self.timelines.get(device_index, []))

    def stop(self):
        self._stop.set()


_SMI = None


def get_amdsmi():
    """Process-wide AMDSmi (None if amdsmi is unavailable)."""
    global _SMI
    if _SMI is None:
        try:
            _SMI = AMDSmi()
            print(f"[Profiler] amdsmi: {len(_SMI.handles)} GPUs, visible {_SMI.visible}")
        except Exception as e:
            print(f"[Profiler] amdsmi energy counters unavailable ({e})")
            _SMI = False
    return _SMI or None


def amd_available():
    """True when running on AMD GPUs through a ROCm PyTorch build."""
    if os.environ.get("ENERGYLENS_BACKEND", "").lower() == "amd":
        return True
    try:
        import torch
        return bool(getattr(torch.version, "hip", None)) and torch.cuda.is_available()
    except Exception:
        return False
