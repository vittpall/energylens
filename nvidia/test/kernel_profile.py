#!/usr/bin/env python3
"""Where the GPU time goes for FP16, bitsandbytes NF4 and AWQ 4-bit (decode-heavy batch).

Each variant runs in its own process (one engine at a time on GPU 0): a warm-up batch,
then one profiled batch of 16 requests with ~70-token prompts and 256 forced output
tokens, through vLLM's torch profiler (engine option profiler_config). The CUDA kernels of
the trace are summed and grouped: GEMM, dequantization, attention, other.

    python test/kernel_profile.py [--out results_figures/kernel_profile]
"""
import argparse
import glob
import gzip
import json
import os
import subprocess
import sys
import time

VARIANTS = {
    "fp16": dict(model="meta-llama/Llama-3.1-8B-Instruct"),
    "bnb4bit": dict(model="meta-llama/Llama-3.1-8B-Instruct", quantization="bitsandbytes"),
    "awq4bit": dict(model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"),
}
GROUPS = [  # first match wins
    ("dequantize", ("dequant", "kdequantize", "bnb", "nf4", "4bit", "unpack")),
    ("gemm (low-bit fused)", ("marlin", "awq", "gptq", "exllama", "w4a16", "int4")),
    ("gemm", ("gemm", "gemv", "cutlass", "sm80_xmma", "ampere_", "cublas", "matmul")),
    ("attention", ("flash", "attn", "attention", "paged")),
]


def child(name, out):
    from vllm import LLM, SamplingParams
    from vllm.config import ProfilerConfig      # vLLM >= 0.16: profiling is an engine option
    prof = ProfilerConfig(profiler="torch", torch_profiler_dir=os.path.join(out, name),
                          torch_profiler_with_stack=False)
    llm = LLM(dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096, enforce_eager=False,
              profiler_config=prof, **VARIANTS[name])
    prompts = ["Summarise the history of the printing press in Europe, its inventors and its "
               "consequences for literacy, religion and science. " * 2] * 16
    sp = SamplingParams(max_tokens=256, min_tokens=256, ignore_eos=True, temperature=0.0)
    llm.generate(prompts, sp)                                        # warm-up
    t = time.perf_counter()
    llm.start_profile()
    llm.generate(prompts, sp)
    llm.stop_profile()
    print(json.dumps({"variant": name, "batch_s": time.perf_counter() - t}), flush=True)
    time.sleep(10)                                                   # let the trace be written


def summarise(out):
    rows = {}
    for name in VARIANTS:
        files = glob.glob(os.path.join(out, name, "**", "*.json*"), recursive=True)
        tot = {}
        for f in files:
            op = gzip.open if f.endswith(".gz") else open
            try:
                ev = json.load(op(f, "rt")).get("traceEvents", [])
            except Exception:
                continue
            for e in ev:
                if e.get("cat") != "kernel":
                    continue
                n = e.get("name", "").lower()
                g = next((g for g, keys in GROUPS if any(k in n for k in keys)), "other")
                tot[g] = tot.get(g, 0.0) + e.get("dur", 0) / 1e3          # ms
                tot["_top_" + e.get("name", "")[:80]] = tot.get("_top_" + e.get("name", "")[:80], 0) + e.get("dur", 0) / 1e3
        rows[name] = tot
    summary = {}
    for name, tot in rows.items():
        groups = {k: v for k, v in tot.items() if not k.startswith("_top_")}
        total = sum(groups.values()) or 1.0
        top = sorted(((v, k[5:]) for k, v in tot.items() if k.startswith("_top_")), reverse=True)[:8]
        summary[name] = {"kernel_ms": round(total, 1),
                         "share_%": {k: round(100 * v / total, 1) for k, v in sorted(groups.items())},
                         "top_kernels_ms": [(round(v, 1), k) for v, k in top]}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="results_figures/kernel_profile")
    ap.add_argument("--child", default=None)
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    if a.child:
        return child(a.child, a.out)
    os.makedirs(a.out, exist_ok=True)
    for name in VARIANTS:
        print(f"== {name}", flush=True)
        subprocess.run([sys.executable, __file__, "--child", name, "--out", a.out], check=False)
    summarise(a.out)


if __name__ == "__main__":
    main()
