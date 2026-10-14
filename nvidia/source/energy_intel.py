"""GPU energy on Intel Data Center GPU Max (Aurora) from the Level Zero energy counters.

Zeus (used on NVIDIA, source/profiler.py) has no Intel GPU backend. Its NVIDIA
method -- read a hardware energy accumulator at the start and end of a window,
on every GPU the engine uses -- is reproduced here with the Level Zero System
Resource Management API (Sysman): ``zesPowerGetEnergyCounter`` returns a
monotonically increasing energy count in microjoules with a microsecond
timestamp, per power domain.

Device hierarchies. A GPU Max 1550 card has two tiles (stacks). Level Zero can
expose them in two ways, selected by ``ZE_FLAT_DEVICE_HIERARCHY``:
  FLAT       (Aurora's default) each tile is a device: 12 per node
  COMPOSITE  each card is a device: 6 per node
Sysman always enumerates *cards*; each card has a card-level power domain
(``onSubdevice == False``) and, on this hardware, one domain per tile
(``onSubdevice == True``, ``subdeviceId`` = tile). A compute device is mapped to
the narrowest domain that covers it:
  FLAT       tile i -> card i // tiles_per_card, tile domain i % tiles_per_card
             (card domain if the driver exposes no tile domains; then a card is
             charged once even if both of its tiles are in use, and a single
             tile is charged with its idle sibling: energy_scope = "card")
  COMPOSITE  card i -> card domain
``ZE_AFFINITY_MASK`` (e.g. "0,1" or "0.0,0.1") is honoured when it restricts the
visible devices. The hierarchy is detected from the number of devices PyTorch
sees versus the cards Sysman reports; ``ENERGYLENS_XPU_HIERARCHY=flat|composite``
overrides it. Run ``python test_intel_energy.py`` on a compute node first: it
checks that loading device i raises the counter that device i maps to.

Power. Sysman has no instantaneous power reading, only the energy counter, so
the background poller derives power from counter differences every 100 ms.
Unlike on NVIDIA (independent power sensor), ``avg_power_watts`` is therefore not
independent of the energy column on Intel (power_source = "ze_counter_poll").
"""
import ctypes
import os
import threading
import time

# Sysman must be enabled before Level Zero is first initialised in the process
# (torch.xpu / vLLM do that); zesInit (newer loaders) does not need it but the
# zeInit fallback below does. Harmless if already set.
os.environ.setdefault("ZES_ENABLE_SYSMAN", "1")

ZE_RESULT_SUCCESS = 0
ZES_STRUCTURE_TYPE_POWER_PROPERTIES = 0xD


class _PowerProps(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p),
                ("onSubdevice", ctypes.c_uint8), ("subdeviceId", ctypes.c_uint32),
                ("canControl", ctypes.c_uint8), ("isEnergyThresholdSupported", ctypes.c_uint8),
                ("defaultLimit", ctypes.c_int32), ("minLimit", ctypes.c_int32),
                ("maxLimit", ctypes.c_int32)]


class _EnergyCounter(ctypes.Structure):
    _fields_ = [("energy", ctypes.c_uint64), ("timestamp", ctypes.c_uint64)]


class LevelZeroError(RuntimeError):
    pass


def _check(rc, what):
    if rc != ZE_RESULT_SUCCESS:
        raise LevelZeroError(f"{what} failed (ze_result_t 0x{rc:x})")


class _Domain:
    """One Sysman power domain: a card, or one tile of a card."""

    def __init__(self, lib, handle, card, tile):
        self.lib, self.handle, self.card, self.tile = lib, handle, card, tile

    @property
    def name(self):
        return f"card{self.card}" + ("" if self.tile is None else f".tile{self.tile}")

    def read(self):
        """(energy J, timestamp s) of the hardware counter."""
        c = _EnergyCounter()
        _check(self.lib.zesPowerGetEnergyCounter(self.handle, ctypes.byref(c)), "zesPowerGetEnergyCounter")
        return c.energy * 1e-6, c.timestamp * 1e-6


class LevelZeroEnergy:
    """Enumerates cards and power domains and maps compute devices to domains."""

    def __init__(self, n_compute_devices=None):
        self.lib = self._load()
        self.cards = self._enum_cards()
        self.domains = []                     # all domains, card and tile
        self.card_domain, self.tile_domains = {}, {}
        for ci, dev in enumerate(self.cards):
            for d in self._enum_domains(dev, ci):
                self.domains.append(d)
                if d.tile is None:
                    self.card_domain[ci] = d
                else:
                    self.tile_domains.setdefault(ci, {})[d.tile] = d
        self.tiles_per_card = max((len(v) for v in self.tile_domains.values()), default=0) or 2
        self.hierarchy, self.visible = self._detect(n_compute_devices)

    # -- Level Zero plumbing -------------------------------------------------
    @staticmethod
    def _load():
        for name in ("libze_loader.so.1", "libze_loader.so"):
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
        raise LevelZeroError("libze_loader.so not found (load the Level Zero / oneAPI module)")

    def _enum_cards(self):
        lib = self.lib
        count = ctypes.c_uint32(0)
        drivers = None
        if hasattr(lib, "zesInit") and lib.zesInit(ctypes.c_uint32(0)) == ZE_RESULT_SUCCESS:
            _check(lib.zesDriverGet(ctypes.byref(count), None), "zesDriverGet")
            drivers = (ctypes.c_void_p * count.value)()
            _check(lib.zesDriverGet(ctypes.byref(count), drivers), "zesDriverGet")
            get_dev = lib.zesDeviceGet
        else:   # older loader: Sysman through core handles (needs ZES_ENABLE_SYSMAN=1)
            _check(lib.zeInit(ctypes.c_uint32(0)), "zeInit")
            _check(lib.zeDriverGet(ctypes.byref(count), None), "zeDriverGet")
            drivers = (ctypes.c_void_p * count.value)()
            _check(lib.zeDriverGet(ctypes.byref(count), drivers), "zeDriverGet")
            get_dev = lib.zeDeviceGet
        cards = []
        for drv in drivers:
            n = ctypes.c_uint32(0)
            _check(get_dev(ctypes.c_void_p(drv), ctypes.byref(n), None), "zesDeviceGet")
            devs = (ctypes.c_void_p * n.value)()
            _check(get_dev(ctypes.c_void_p(drv), ctypes.byref(n), devs), "zesDeviceGet")
            cards += [ctypes.c_void_p(d) for d in devs]
        if not cards:
            raise LevelZeroError("Sysman reports no GPUs")
        return cards

    def _enum_domains(self, dev, ci):
        lib = self.lib
        n = ctypes.c_uint32(0)
        _check(lib.zesDeviceEnumPowerDomains(dev, ctypes.byref(n), None), "zesDeviceEnumPowerDomains")
        hs = (ctypes.c_void_p * n.value)()
        _check(lib.zesDeviceEnumPowerDomains(dev, ctypes.byref(n), hs), "zesDeviceEnumPowerDomains")
        out = []
        for h in hs:
            p = _PowerProps(stype=ZES_STRUCTURE_TYPE_POWER_PROPERTIES)
            _check(lib.zesPowerGetProperties(ctypes.c_void_p(h), ctypes.byref(p)), "zesPowerGetProperties")
            out.append(_Domain(lib, ctypes.c_void_p(h), ci, int(p.subdeviceId) if p.onSubdevice else None))
        if not any(d.tile is None for d in out) and hasattr(lib, "zesDeviceGetCardPowerDomain"):
            h = ctypes.c_void_p()
            if lib.zesDeviceGetCardPowerDomain(dev, ctypes.byref(h)) == ZE_RESULT_SUCCESS:
                out.append(_Domain(lib, h, ci, None))
        return out

    # -- hierarchy and device mapping ------------------------------------------
    def _all_units(self, hierarchy):
        if hierarchy == "flat":
            return [(c, t) for c in range(len(self.cards)) for t in range(self.tiles_per_card)]
        return [(c, None) for c in range(len(self.cards))]

    def _detect(self, n_compute):
        n_cards = len(self.cards)
        forced = os.environ.get("ENERGYLENS_XPU_HIERARCHY", "").lower()
        env = os.environ.get("ZE_FLAT_DEVICE_HIERARCHY", "").upper()
        mask = [m for m in os.environ.get("ZE_AFFINITY_MASK", "").split(",") if m.strip()]
        if forced in ("flat", "composite"):
            hierarchy = forced
        elif env in ("FLAT", "COMPOSITE"):
            hierarchy = env.lower()
        elif n_compute is not None and n_compute > n_cards:
            hierarchy = "flat"
        elif n_compute is not None and n_compute == n_cards and not mask:
            hierarchy = "composite"
        else:
            hierarchy = "flat"          # Aurora's default
        units = self._all_units(hierarchy)
        if mask:                        # visible devices restricted by the mask
            vis = []
            for m in mask:
                m = m.strip()
                if "." in m:            # card.tile
                    c, t = (int(x) for x in m.split(".", 1))
                    vis.append((c, t) if hierarchy == "flat" else (c, None))
                elif hierarchy == "flat":
                    vis.append(units[int(m)])
                else:
                    vis.append((int(m), None))
            units = vis
        return hierarchy, units

    def domains_for(self, device_indices):
        """Distinct domains covering the given compute devices, and the scope used."""
        chosen, scope = [], "tile" if self.hierarchy == "flat" else "card"
        for i in device_indices:
            c, t = self.visible[i]
            if t is not None and t in self.tile_domains.get(c, {}):
                d = self.tile_domains[c][t]
            else:
                d = self.card_domain.get(c)
                if d is None:
                    raise LevelZeroError(f"no power domain for card {c}")
                if t is not None:
                    scope = "card"
            if d not in chosen:
                chosen.append(d)
        return chosen, scope

    def describe(self):
        tiles = {c: sorted(v) for c, v in self.tile_domains.items()}
        return (f"{len(self.cards)} cards, tile domains {tiles or 'none'}, hierarchy {self.hierarchy}, "
                f"{len(self.visible)} visible devices -> {self.visible}")


class _Measurement:
    def __init__(self, t, per_device):
        self.time = t
        self.gpu_energy = per_device
        self.total_energy = sum(per_device.values())


class IntelEnergyMonitor:
    """Zeus-like window API: begin_window / end_window over a set of compute devices.

    gpu_energy is keyed by compute-device index; when several devices share one
    card domain (FLAT without tile domains), the card's energy is attributed to
    the first of them so that nothing is counted twice."""

    def __init__(self, ze, device_indices):
        self.ze = ze
        self.gpu_indices = list(device_indices)
        self.domains, self.scope = ze.domains_for(self.gpu_indices)
        self._owner = {}
        for i in self.gpu_indices:
            d, _ = ze.domains_for([i])
            self._owner.setdefault(d[0].name, i)
        self._open = {}

    def _read_all(self):
        return {d.name: d.read() for d in self.domains}

    def begin_window(self, key):
        self._open[key] = (time.time(), self._read_all())

    def end_window(self, key, cancel=False):
        t0, start = self._open.pop(key)
        if cancel:
            return None
        end = self._read_all()
        per_dev = {}
        for d in self.domains:
            e0, _ = start[d.name]
            e1, _ = end[d.name]
            per_dev[self._owner[d.name]] = max(e1 - e0, 0.0)
        return _Measurement(time.time() - t0, per_dev)


class IntelPowerPoller:
    """Background thread: power per domain from counter differences, every period_s."""

    def __init__(self, ze, period_s=0.1, cap=8192):
        self.ze, self.period_s, self.cap = ze, period_s, cap
        self.update_period = period_s
        self.timelines = {d.name: [] for d in ze.domains}
        self._last = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            now = time.time()
            for d in self.ze.domains:
                try:
                    e, ts = d.read()
                except Exception:
                    continue
                prev = self._last.get(d.name)
                self._last[d.name] = (e, ts)
                if prev and ts > prev[1]:
                    w = (e - prev[0]) / (ts - prev[1])
                    with self._lock:
                        tl = self.timelines[d.name]
                        tl.append((now, w))
                        if len(tl) > self.cap:
                            del tl[: len(tl) - self.cap]
            self._stop.wait(self.period_s)

    def timeline(self, domain_name):
        with self._lock:
            return list(self.timelines.get(domain_name, []))

    def stop(self):
        self._stop.set()


_ZE = None


def get_level_zero(n_compute_devices=None):
    """Process-wide LevelZeroEnergy (None if Level Zero / Sysman is unavailable)."""
    global _ZE
    if _ZE is None:
        try:
            _ZE = LevelZeroEnergy(n_compute_devices)
            print(f"[Profiler] Level Zero Sysman: {_ZE.describe()}")
        except Exception as e:
            print(f"[Profiler] Level Zero energy counters unavailable ({e})")
            _ZE = False
    return _ZE or None


def xpu_available():
    """True when running on Intel GPUs through PyTorch XPU (and not on CUDA)."""
    if os.environ.get("ENERGYLENS_BACKEND", "").lower() == "intel":
        return True
    try:
        import torch
        return (not torch.cuda.is_available()) and hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception:
        return False


def xpu_device_count():
    try:
        import torch
        return torch.xpu.device_count()
    except Exception:
        return None
