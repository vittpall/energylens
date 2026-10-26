# AMD Instinct MI100 backend

Energy on AMD GPUs comes from the amdsmi hardware energy accumulator
(`source/energy_amd.py`), read at the start and end of each generation window and summed
over every GPU the engine occupies, through the same `begin_window` / `end_window`
interface as Zeus (NVIDIA) and Level Zero (Intel). Power is sampled separately from the
firmware socket-power sensor every 100 ms (`power_source = amdsmi_sampling`). The CSV
format is unchanged; `validate_results.py` recognises the data as `MI100`.

Zeus 0.16 has an AMD backend, but it needs a newer amdsmi
(`amdsmi_get_gpu_enumeration_info`) than the container has, and it integrates polled power
for the energy of a window instead of reading the counter.

## Counter scale
On MI100, `amdsmi_get_energy_count` reports `counter_resolution = 15.3` (uJ), but the
accumulator advances by about 1 J per count (15.3 uJ x 2^16). A 5 s GEMM load at 279 W
moved it by 1413 counts. `ENERGYLENS_AMD_COUNTER_SCALE` (default 65536) holds the factor.
`python test_amd_energy.py` loads each GPU in turn. For the loaded GPU, it checks that the
counter difference matches sampled power x time; run it before every profiling job.
rocm-smi's "Accumulated Energy (uJ)" has the same 2^16 error.

## Container
Build the image from `docker/Dockerfile.rocm-mi100` (ROCm PyTorch base, amdsmi, build tools), then
compile vLLM 0.6.3 inside it once:
```
docker build -f docker/Dockerfile.rocm-mi100 -t energylens-rocm .
docker run -d --name vllm --pid=host --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 32g --security-opt seccomp=unconfined -v ~/work:/work \
  energylens-rocm sleep infinity
docker exec vllm bash /work/<repo>/nvidia/amd_setup/build_vllm_mi100.sh   # once, builds vLLM 0.6.3 for gfx908
```
`--pid=host` is needed by the GPU guard (`source/gpu_guard.py`). amdsmi reports host PIDs,
and without it the profiler's own process looks like another job's.

`build_vllm_mi100.sh` builds vLLM under `/work/build/vllm` (the Dockerfile sets
`PYTHONPATH=/work/build/vllm`, so a new container finds it again).

## Differences from the A100 / Max 1550 runs
- fp16 (MI100 has no bfloat16).
- Layouts (TP, PP) = (1, 1) and (2, 1) only. vLLM 0.6.3's synchronous `LLM` engine
  rejects PP > 1.
- Eager mode on every layout (`enforce_eager`, set in `source/model.py` on AMD). With
  captured graphs, TP=2 returns the right first token and garbage afterwards. In eager
  mode its output matches TP=1 token for token (`check_layout_output.py MODEL 2 1`, with
  `EAGER=1`).
- `NCCL_P2P_DISABLE=1`: RCCL peer-to-peer between the two PCIe root complexes fails.
- The ROCm custom paged-attention kernel aborts on gfx908, so it is disabled in the build.
- Chunked prefill is enabled explicitly (V0 engine default is off; the budget sweep needs it).
- Qwen2.5-14B fp16 does not fit on one 32 GB GPU: TP=2 only (a single layout).

Run everything with `amd_setup/run_amd.sh`; output goes to `results_amd/`.
