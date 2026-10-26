# Setting up the profiling environment

The analysis scripts (fitting, baselines, figures) run on a CPU with `pip install -r requirements.txt`.
This file is for re-running the **measurements**. The profiler is the same on every vendor
(`nvidia/main_profiler.py`, same grid, same CSV format); only the engine build and the energy backend
differ. Each backend reads a hardware energy counter over the generation window, summed over every GPU of
the engine, and samples power separately.

| | NVIDIA A100 | Intel Data Center GPU Max 1550 | AMD Instinct MI100 |
|---|---|---|---|
| vLLM | 0.16 | 0.18, XPU build + `intel/vllm_xpu_patches.diff` | 0.6.3, built for gfx908 |
| energy source | Zeus / NVML counter | Level Zero Sysman counter | amdsmi accumulator |
| `ENERGYLENS_BACKEND` | (unset) | `intel` | `amd` |
| dtype | bf16 / fp16 | bf16 | fp16 (no bf16 on MI100) |
| layouts profiled | TP x PP up to 8 GPUs | TP x PP up to 12 tiles | (TP, PP) = (1,1), (2,1) |
| self-test | n/a | `intel/test_intel_energy.py` | `nvidia/test_amd_energy.py` |

After any run, `python nvidia/validate_results.py <profiling_metrics.csv>` flags the rows to drop (energy
that disagrees with power x latency, failed or truncated generations). The scripts in `nvidia/test/` apply
it on load.

## NVIDIA

```bash
python -m venv env && source env/bin/activate
pip install -r requirements-profiler-nvidia.txt      # vLLM 0.16, PyTorch 2.9, NVML bindings
pip install zeus-ml                                  # energy counter (falls back to a two-point NVML estimate without it)
cd nvidia
python main_profiler.py --model mistralai/Mistral-7B-Instruct-v0.3 --tp 1 2 4 --pp 1 2 4 --output-root ./results_zeus
```

The job needs exclusive use of its GPUs: the profiler refuses to start if another process runs on them
(`source/gpu_guard.py`). A run that stops resumes when the same command is started again.
`--min-gpus 8` with `--tp 1 2 4 8 --pp 1 2 4 8` profiles only the 8-GPU layouts.

## Intel Data Center GPU Max 1550

1. A PyTorch XPU environment (oneAPI runtime, `torch.xpu.is_available()` true) with vLLM built for XPU.
2. Apply the patches to the vLLM source tree, then reinstall it:
   ```bash
   cd vllm && git apply /path/to/intel/vllm_xpu_patches.diff && pip install -e . --no-build-isolation
   ```
3. Environment variables (set before the first Level Zero call):
   ```bash
   export ENERGYLENS_BACKEND=intel
   export ZES_ENABLE_SYSMAN=1                  # Sysman energy counters
   export ZE_FLAT_DEVICE_HIERARCHY=FLAT        # one device per tile (COMPOSITE: one per card)
   export TORCHDYNAMO_DISABLE=1
   ```
   On multi-rank runs, oneCCL needed `CCL_PROCESS_LAUNCHER=none CCL_ATL_TRANSPORT=ofi FI_PROVIDER=tcp`.
4. Self-test (about a minute; do not profile if it fails), then profile:
   ```bash
   cd intel
   python test_intel_energy.py --seconds 4
   python main_profiler.py --model meta-llama/Llama-3.1-8B-Instruct --tp 1 2 4 --pp 1 2 3 4 6 --output-root ./results_intel
   ```
   `--tp` / `--pp` count devices (tiles in FLAT mode). Without tile power domains a one-tile layout is
   charged its whole card (`energy_scope = card` in the CSV).

## AMD Instinct MI100

Needs a host with the ROCm kernel driver and `/dev/kfd`, and Docker. vLLM 0.6.3 is built for gfx908 inside a
container, so nothing but the driver is installed on the host.

```bash
docker build -f docker/Dockerfile.rocm-mi100 -t energylens-rocm .
docker run -d --name vllm --pid=host --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 32g --security-opt seccomp=unconfined -v "$PWD/..":/work \
  energylens-rocm sleep infinity
docker exec vllm bash /work/<this-repo>/nvidia/amd_setup/build_vllm_mi100.sh     # once: compiles vLLM 0.6.3
docker exec vllm bash -c "cd /work/<this-repo>/nvidia && python test_amd_energy.py"   # counter self-test
docker exec -d vllm bash /work/<this-repo>/nvidia/amd_setup/run_amd.sh                 # Mistral-7B, Llama-3.1-8B, Qwen2.5-14B
```

Notes (details in `nvidia/amd_setup/README_amd.md`):
- The Dockerfile takes the base image as `--build-arg BASE_IMAGE=...` (a ROCm PyTorch image). The paper's runs used ROCm 6.1 and
  PyTorch 2.4.0 (rocm6.1, Python 3.10); pass the matching official tag and pin it.
- `--pid=host` is required by the GPU guard.
- The accumulator advances in units of 15.3 uJ x 2^16; `ENERGYLENS_AMD_COUNTER_SCALE` (default 65536) holds
  the factor, and `test_amd_energy.py` checks it against sampled power x time before every job.
- Layouts are (1,1) and (2,1): vLLM 0.6.3's synchronous engine has no pipeline parallelism, and the node has
  two GPUs. Qwen2.5-14B in fp16 does not fit one 32 GB GPU.
- Evaluation of the resulting data: `python nvidia/test/amd_eval.py`, `python nvidia/viz/amd.py`.
