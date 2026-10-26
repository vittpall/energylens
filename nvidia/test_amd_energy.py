"""Check the amdsmi energy counter on AMD GPUs: load device i, compare the counter
difference with the sampled socket power x time. Run inside the ROCm container."""
import sys, time
import torch
from source.energy_amd import get_amdsmi, AMDEnergyMonitor, AMDPowerPoller

smi = get_amdsmi()
poller = AMDPowerPoller(smi)
bad = False
for dev in range(len(smi.visible)):
    x = torch.randn(8192, 8192, device=f"cuda:{dev}", dtype=torch.float16)
    torch.cuda.synchronize(dev)
    mon = AMDEnergyMonitor(smi, list(range(len(smi.visible))))
    mon.begin_window("w"); t0 = time.time()
    while time.time() - t0 < 5:
        for _ in range(50):
            x @ x
        torch.cuda.synchronize(dev)
    t1 = time.time(); m = mon.end_window("w")
    p = [sum(w for ts, w in poller.timeline(i) if t0 <= ts <= t1) / max(1, len([1 for ts, _ in poller.timeline(i) if t0 <= ts <= t1])) for i in range(len(smi.visible))]
    print(f"loaded cuda:{dev}: counter J {m.gpu_energy}, window {m.time:.2f}s, sampled W {p}, "
          f"power x time J {[round(w * m.time, 1) for w in p]}")
    ratio = m.gpu_energy[dev] / (p[dev] * m.time)
    if not 0.9 < ratio < 1.1:
        bad = True
        print(f"  MISMATCH on cuda:{dev}: counter / (power x time) = {ratio:.3f}")
sys.exit(1 if bad else 0)
