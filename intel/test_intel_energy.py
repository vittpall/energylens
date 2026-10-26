#!/usr/bin/env python3
"""Self-test of the Intel GPU energy measurement (source/energy_intel.py). Run it on an
Intel GPU compute node before any profiling job:

    python test_intel_energy.py                # every visible device, 4 s of load each
    python test_intel_energy.py --devices 0 1  # a subset

It checks, and prints PASS/FAIL for:
  1. Level Zero Sysman is available and reports power domains (tile and/or card level);
  2. the device hierarchy (FLAT tiles or COMPOSITE cards) and the device -> domain map;
  3. for every tested device, loading that device raises the energy counter of the
     domain it is mapped to more than any other domain (the map is right);
  4. the energy of a window agrees with the polled power x time (within 10%), and the
     loaded power is plausible (above idle, below 1.2 x the card TDP of 600 W).
A FAIL in 3 means the device order of Sysman and PyTorch differ: report the printed
table, or set ENERGYLENS_XPU_HIERARCHY / ZE_AFFINITY_MASK accordingly.
"""
import os
os.environ.setdefault("ZES_ENABLE_SYSMAN", "1")   # before torch initialises Level Zero

import argparse
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
from source.energy_intel import LevelZeroEnergy, IntelEnergyMonitor, IntelPowerPoller  # noqa: E402

CARD_TDP_W = 600.0


def load(device, seconds):
    a = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    torch.xpu.synchronize(device)
    t = time.time()
    while time.time() - t < seconds:
        for _ in range(10):
            a = (a @ b).clamp_(-1, 1)
        torch.xpu.synchronize(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--devices", type=int, nargs="+", default=None)
    ap.add_argument("--seconds", type=float, default=4.0)
    args = ap.parse_args()
    ok = True

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        sys.exit("FAIL: torch.xpu is not available (load the frameworks module / activate the XPU env)")
    n = torch.xpu.device_count()
    ze = LevelZeroEnergy(n)
    print(f"torch.xpu devices: {n}")
    print(f"Sysman: {ze.describe()}")
    print(f"ZE_FLAT_DEVICE_HIERARCHY={os.environ.get('ZE_FLAT_DEVICE_HIERARCHY', 'unset')}  "
          f"ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK', 'unset')}")
    has_tiles = bool(ze.tile_domains)
    print(f"[{'PASS' if ze.domains else 'FAIL'}] 1. power domains: {[d.name for d in ze.domains]}")
    ok &= bool(ze.domains)
    if ze.hierarchy == "flat" and not has_tiles:
        print("[WARN] FLAT hierarchy but no tile power domains: single-tile runs will be charged the whole card")
    print(f"[INFO] 2. hierarchy {ze.hierarchy}; device -> domain: "
          + ", ".join(f"{i}->{ze.domains_for([i])[0][0].name}" for i in range(len(ze.visible))))

    poller = IntelPowerPoller(ze, period_s=0.1)
    time.sleep(1.5)
    idle = {}
    for d in ze.domains:
        tl = [w for _, w in poller.timeline(d.name)]
        idle[d.name] = sum(tl) / len(tl) if tl else float("nan")
    print("idle power (W): " + ", ".join(f"{k} {v:.0f}" for k, v in idle.items()))

    devices = args.devices if args.devices is not None else list(range(min(n, len(ze.visible))))
    for i in devices:
        start = {d.name: d.read()[0] for d in ze.domains}
        mon = IntelEnergyMonitor(ze, [i])
        mon.begin_window("w")
        t0 = time.time()
        load(f"xpu:{i}", args.seconds)
        t1 = time.time()
        m = mon.end_window("w")
        delta = {d.name: d.read()[0] - start[d.name] for d in ze.domains}
        # compare like with like: the narrowest domains (tiles if present)
        cand = {k: v for k, v in delta.items() if (".tile" in k) == (mon.scope == "tile")}
        hottest = max(cand, key=lambda k: cand[k] - idle.get(k, 0) * (t1 - t0))
        mapped = mon.domains[0].name
        good3 = hottest == mapped
        poll = sum(sum(w for ts, w in poller.timeline(d.name) if t0 <= ts <= t1) /
                   max(1, sum(1 for ts, _ in poller.timeline(d.name) if t0 <= ts <= t1))
                   for d in mon.domains)
        window_w = m.total_energy / m.time
        agree = abs(window_w - poll) / max(window_w, 1e-9) < 0.10
        plausible = idle.get(mapped, 0) < window_w < 1.2 * CARD_TDP_W
        print(f"device {i}: mapped {mapped}, hottest {hottest} -> [{'PASS' if good3 else 'FAIL'}] 3; "
              f"window {m.total_energy:.0f} J / {m.time:.2f} s = {window_w:.0f} W, polled {poll:.0f} W "
              f"-> [{'PASS' if agree and plausible else 'FAIL'}] 4")
        ok &= good3 and agree and plausible
    poller.stop()
    print("ALL PASS" if ok else "SOME CHECKS FAILED -- do not trust the energy column yet")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
