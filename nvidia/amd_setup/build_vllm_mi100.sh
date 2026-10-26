#!/bin/bash
# Build vLLM 0.6.3 for AMD Instinct MI100 (gfx908) inside the container built from docker/Dockerfile.rocm-mi100.
# The image ships an editable vLLM whose source dir (/workspace/vllm) is empty, so vLLM is
# rebuilt from source. Run inside the container (docker run ... --pid=host, see README_amd.md).
set -e
mkdir -p /work/build && cd /work/build
[ -d vllm ] || git clone --depth 1 --branch v0.6.3 https://github.com/vllm-project/vllm.git
cd vllm
# The ROCm custom paged-attention kernel (csrc/rocm/attention.hip) aborts with a hardware
# exception on gfx908; use the Triton/fallback path instead.
python - <<'PY'
p = "vllm/attention/backends/rocm_flash_attn.py"
s = open(p).read()
a = "    return (not _ON_NAVI and"
if a in s:
    s = s.replace(a, "    return (False and not _ON_NAVI and")   # gfx908: custom kernel crashes
    open(p, "w").write(s)
PY
grep -v -E "^(torch|torchvision|xformers)" requirements-common.txt > /tmp/req.txt
pip install -q setuptools-scm jinja2 -r /tmp/req.txt "numpy<2" pandas datasets
# sccache in the image can hang the build: take it off PATH
export PATH=$(echo $PATH | tr ":" "\n" | grep -v /opt/cache | paste -sd:)
PYTORCH_ROCM_ARCH=gfx908 VLLM_TARGET_DEVICE=rocm MAX_JOBS=32 pip install --no-build-isolation --no-deps -e .
