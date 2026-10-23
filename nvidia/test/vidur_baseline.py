#!/usr/bin/env python3
"""Vidur baseline: simulate every grid batch with Vidur and record its latency.

Vidur (Agrawal et al., MLSys 2024; github.com/microsoft/vidur) predicts the
latency of a serving configuration by simulating the scheduler over per-operator
execution times that it profiled itself; MaverIQ uses it as a baseline. It ships
A100 operator profiles for Meta-Llama-3-8B (same architecture as Llama-3.1-8B),
so no profiling of ours is involved: the prediction is zero-shot.

For each distinct grid batch (TP, PP, token budget, batch size, prompt tokens
per request, decode tokens) we replay the batch through Vidur's vLLM scheduler,
all requests arriving at t=0, with our concurrency cap (64) and budget, and take
the time until the last request finishes as the batch latency. Every request
gets the batch's mean prompt length and its longest output (max_generated_tokens),
the decode length the batch runs for. Energy then follows the MaverIQ adaptation
of the paper (latency x the configuration's power / batch size), in vidur_eval.py.

    python test/vidur_baseline.py --grid results_zeus/run_text_only_Llama-3.1-8B-Instruct_fp16/res_native/profiling_metrics.csv

--scheduler sarathi models chunked prefill within the token budget, as the vLLM V1 engine
we profile does (Vidur's "vllm" scheduler is the older prefill-first vLLM);
--network a100_pairwise_nvlink is Vidur's interconnect profile without a full NVLink switch.
--model and --tp select another of Vidur's bundled A100 profiles.
"""
import argparse
import glob
import hashlib
import multiprocessing as mp
import os
import subprocess

import pandas as pd

VIDUR = os.environ.get("VIDUR_DIR", "../vidur")                 # a clone of github.com/microsoft/vidur
PY = os.environ.get("VIDUR_PYTHON", "python")                   # a Python environment with Vidur installed
RUNS = os.environ.get("VIDUR_RUNS", "./vidur_runs")             # simulation outputs and predictor cache
MODEL = "meta-llama/Meta-Llama-3-8B"
MAX_SEQS = 64
SUPPORTED_TP = (1, 4)    # TP degrees with attention profiles in Vidur's A100 data for this model


def keys(grid):
    g = pd.read_csv(grid)
    g = g[pd.to_numeric(g["energy"], errors="coerce") > 0]
    dec = g["max_generated_tokens"] if "max_generated_tokens" in g else g["max_tokens"]
    k = pd.DataFrame({"tp": g["tensor_parallel_size"].astype(int), "pp": g["pipeline_parallel_size"].astype(int),
                      "budget": g["max_num_batched_tokens"].astype(int), "bs": g["batch_size"].astype(int),
                      "prefill": (g["input_length"] / g["batch_size"]).round().astype(int).clip(lower=1),
                      "decode": dec.fillna(g["max_tokens"]).astype(int).clip(lower=1)})
    return k.drop_duplicates().reset_index(drop=True)


def run_one(row):
    from types import SimpleNamespace
    row = SimpleNamespace(**row)
    sched, net, model = row.scheduler, row.network, row.model
    vals = tuple(v for k, v in vars(row).items() if k not in ("scheduler", "network", "model"))
    extra = "" if (sched, net, model) == ("vllm", "a100_dgx", MODEL) else f"{sched}{net}{model}"
    tag = hashlib.md5((str(vals) + extra).encode()).hexdigest()[:12]
    out = os.path.join(RUNS, "grid", tag)
    done = glob.glob(os.path.join(out, "*", "request_metrics.csv"))
    if not done:
        os.makedirs(out, exist_ok=True)
        trace = os.path.join(out, "trace.csv")
        pd.DataFrame({"arrived_at": 0.0, "num_prefill_tokens": [row.prefill] * row.bs,
                      "num_decode_tokens": [row.decode] * row.bs}).to_csv(trace, index=False)
        cmd = [PY, "-m", "vidur.main", "--replica_config_device", "a100",
               "--replica_config_network_device", net, "--replica_config_model_name", model,
               "--cluster_config_num_replicas", "1",
               "--replica_config_tensor_parallel_size", str(row.tp),
               "--replica_config_num_pipeline_stages", str(row.pp),
               "--request_generator_config_type", "trace_replay",
               "--trace_request_generator_config_trace_file", trace,
               "--trace_request_generator_config_max_tokens", "16384",
               "--replica_scheduler_config_type", sched,
               *([f"--vllm_scheduler_config_batch_size_cap", str(MAX_SEQS),
                  "--vllm_scheduler_config_max_tokens_in_batch", str(row.budget)] if sched == "vllm" else
                 ["--sarathi_scheduler_config_batch_size_cap", str(MAX_SEQS),
                  "--sarathi_scheduler_config_chunk_size", str(row.budget)]),
               "--metrics_config_output_dir", out, "--metrics_config_cache_dir", os.path.join(RUNS, "cache"),
               "--no-metrics_config_enable_chrome_trace", "--no-metrics_config_store_plots",
               # the long-prompt grid exceeds Vidur's default prediction ranges
               "--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size", "65536",
               "--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request", "16384",
               "--random_forrest_execution_time_predictor_config_prediction_max_batch_size", "512"]
        r = subprocess.run(cmd, cwd=VIDUR, capture_output=True, text=True,
                           env={**os.environ, "OMP_NUM_THREADS": "1"})
        done = glob.glob(os.path.join(out, "*", "request_metrics.csv"))
        if not done:
            return {**vars(row), "vidur_latency": float("nan"), "error": r.stderr[-300:]}
    m = pd.read_csv(done[0])
    return {**vars(row), "vidur_latency": float(m["request_e2e_time"].max()), "error": ""}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--grid", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results_figures/vidur/vidur_latency.csv")
    ap.add_argument("--scheduler", default="vllm", choices=["vllm", "sarathi"])
    ap.add_argument("--network", default="a100_dgx")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tp", type=int, nargs="+", default=list(SUPPORTED_TP))
    args = ap.parse_args()
    k = pd.concat([keys(g) for g in args.grid]).drop_duplicates().reset_index(drop=True)
    # Vidur's A100 Llama-3-8B attention profile covers TP 1 and 4 only
    k = k[k["tp"].isin(args.tp)].reset_index(drop=True)
    k = k.assign(scheduler=args.scheduler, network=args.network, model=args.model)
    print(f"{len(k)} distinct batches (TP in {args.tp}), {args.scheduler} scheduler, {args.network}", flush=True)
    # train Vidur's operator-time predictors once per (TP, PP) before fanning out
    first = k.drop_duplicates(["tp", "pp"])
    for r in first.to_dict("records"):
        print("warm-up", run_one(r)["vidur_latency"], r, flush=True)
    with mp.Pool(args.workers) as pool:
        res = pd.DataFrame(pool.map(run_one, k.to_dict("records"), chunksize=4))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"saved {args.out}: {res['vidur_latency'].notna().sum()}/{len(res)} simulated", flush=True)
    if res["error"].astype(bool).any():
        print(res.loc[res["error"].astype(bool), "error"].head(3).to_string())


if __name__ == "__main__":
    main()
