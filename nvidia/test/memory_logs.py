#!/usr/bin/env python3
"""Collect vLLM engine memory figures from the profiling job logs.

vLLM logs, for every engine it starts, the weights loaded on the first worker
(first pipeline stage, TP rank 0), that worker's memory left for the KV cache,
the engine's KV cache size in tokens, and whether start-up finished. This turns
the PBS logs into one row per engine start:

    model, quantization, tp, pp, budget (max_num_batched_tokens), max_seq_len,
    weights_gib, kv_avail_gib, kv_tokens, graph_gib, ok, error

    python test/memory_logs.py                 # -> results_figures/memory/engines.csv
"""
import glob
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NVIDIA = os.path.dirname(HERE)
LOG_DIRS = [os.path.join(NVIDIA, "logs"), os.path.join(NVIDIA, "..", "adaptive_sampling", "logs")]
OUT = os.path.join(NVIDIA, "results_figures", "memory")

R_BUDGET = re.compile(r"max_num_batched_tokens=(\d+)")
R_INIT = re.compile(r"Initializing a V1 LLM engine .*?model='([^']+)'.*?max_seq_len=(\d+).*?"
                    r"tensor_parallel_size=(\d+), pipeline_parallel_size=(\d+).*?quantization=(\w+)")
R_WEIGHTS = re.compile(r"Model loading took ([\d.]+) GiB")
R_AVAIL = re.compile(r"Available KV cache memory: (-?[\d.]+) GiB")
R_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
R_GRAPH = re.compile(r"Graph capturing finished in \d+ secs, took ([\d.]+) GiB")
R_DONE = re.compile(r"init engine \(profile, create kv cache, warmup model\) took")
R_ERR = re.compile(r"(CUDA out of memory|To serve at least one request|larger than the available KV cache"
                   r"|No available memory for the cache blocks|is not divisible by)")


def parse_file(path):
    rows, cur, budget = [], None, None

    def close():
        if cur is not None:
            rows.append(cur)

    with open(path, errors="ignore") as f:
        for line in f:
            m = R_INIT.search(line)
            if m:
                close()
                cur = {"log": os.path.basename(path), "model": m[1], "max_seq_len": int(m[2]),
                       "tp": int(m[3]), "pp": int(m[4]),
                       "quantization": "none" if m[5] == "None" else m[5],
                       "budget": budget, "weights_gib": None, "kv_avail_gib": None,
                       "kv_tokens": None, "graph_gib": None, "ok": False, "error": ""}
                continue
            m = R_BUDGET.search(line)
            if m and "Chunked prefill" in line:
                budget = int(m[1])
            if cur is None:
                continue
            for key, rx, conv in (("weights_gib", R_WEIGHTS, float), ("kv_avail_gib", R_AVAIL, float),
                                  ("graph_gib", R_GRAPH, float),
                                  ("kv_tokens", R_TOKENS, lambda s: int(s.replace(",", "")))):
                m = rx.search(line)
                if m and cur[key] is None:
                    cur[key] = conv(m[1])
            if R_DONE.search(line):
                cur["ok"] = True
                close()
                cur = None
                continue
            m = R_ERR.search(line)
            if m and not cur["error"]:
                cur["error"] = m[1]
    close()
    return rows


def collect():
    rows = []
    for d in LOG_DIRS:
        for p in sorted(glob.glob(os.path.join(d, "*"))):
            if os.path.isfile(p):
                rows += parse_file(p)
    df = pd.DataFrame(rows)
    df = df[~df["model"].str.contains("/fingerprints/")]       # truncated models of the fingerprint study
    return df.reset_index(drop=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df = collect()
    df.to_csv(os.path.join(OUT, "engines.csv"), index=False)
    print(f"{len(df)} engine starts, {df['ok'].sum()} finished start-up -> {OUT}/engines.csv")
    print(df.groupby(["model", "quantization"]).agg(starts=("ok", "size"), ok=("ok", "sum"),
                                                    layouts=("tp", lambda s: s.nunique())).to_string())
