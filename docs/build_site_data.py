#!/usr/bin/env python3
"""Export the compact JSON the project page reads (docs/data/site.json).

    python docs/build_site_data.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "adaptive_sampling"))
sys.path.insert(0, os.path.join(ROOT, "nvidia"))
from simulator import load_csv  # noqa: E402
from compare_cross_model import fit_unified, unified_predict  # noqa: E402
from viz.datasets import DATASETS  # noqa: E402

Z = "results_zeus/run_text_only_{}/res_native/profiling_metrics.csv"
QUANT = [("Llama-3.1-8B", "awq4bit", "Meta-Llama-3.1-8B-Instruct-AWQ-INT4_awq4bit"),
         ("Llama-3.1-8B", "w8a8", "Meta-Llama-3.1-8B-Instruct-quantized.w8a8_w8a8"),
         ("Llama-3.1-8B", "bnb4bit", "Llama-3.1-8B-Instruct_bnb4bit"),
         ("Qwen2.5-14B", "awq4bit", "Qwen2.5-14B-Instruct-AWQ_awq4bit"),
         ("Qwen2.5-14B", "gptq8bit", "Qwen2.5-14B-Instruct-GPTQ-Int8_gptq8bit")]
AURORA = [("Mistral-7B", "text", "text_only_Mistral-7B-Instruct-v0.3_fp16"),
          ("Llama-3.1-8B", "text", "text_only_Llama-3.1-8B-Instruct_fp16"),
          ("Qwen2.5-14B", "text", "text_only_Qwen2.5-14B-Instruct_fp16"),
          ("Qwen1.5-MoE", "text", "text_only_Qwen1.5-MoE-A2.7B_fp16"),
          ("Qwen2-VL-7B", "image", "text_image_Qwen2-VL-7B-Instruct_fp16"),
          ("Mamba-Codestral-7B", "text", "text_only_Mamba-Codestral-7B-v0.1_fp16"),
          ("Llama-3.1-8B", "text", "text_only_Llama-3.1-8B-Instruct_bnb4bit", "bnb4bit"),
          ("Llama-3.1-8B", "text", "text_only_Meta-Llama-3.1-8B-Instruct-AWQ-INT4_awq4bit", "awq4bit"),
          ("Qwen2.5-14B", "text", "text_only_Qwen2.5-14B-Instruct_bnb4bit", "bnb4bit"),
          ("Qwen2.5-14B", "text", "text_only_Qwen2.5-14B-Instruct-AWQ_awq4bit", "awq4bit")]
INLEN = [("Llama-3.1-8B", "Llama-3.1-8B-Instruct"), ("Qwen2.5-14B", "Qwen2.5-14B-Instruct")]
KEYS = ["max_tokens", "batch_size", "tp_degree", "pp_degree", "budget"]


def cells(df, predict):
    d = df.assign(pred=unified_predict(fit_unified(df.sample(min(50, len(df)), random_state=42)), df)
                  if predict else np.nan)
    g = d.groupby(KEYS).agg(E=("measured_energy", "mean"), L=("measured_latency", "mean"),
                            P=("gpu_power_w", "mean"), Ep=("pred", "mean")).reset_index()
    rows = []
    for r in g.itertuples(index=False):
        rows.append([int(r.max_tokens), int(r.batch_size), int(r.tp_degree), int(r.pp_degree), int(r.budget),
                     round(r.E, 3), round(r.L, 4), round(r.P, 1), None if np.isnan(r.Ep) else round(r.Ep, 3)])
    return rows


def entry(hw, model, modality, precision, path, predict):
    df = load_csv(path)
    print(f"{hw:9s} {model:18s} {modality:10s} {precision:8s} {len(df):6d} rows")
    return {"hardware": hw, "model": model, "modality": modality, "precision": precision,
            "rows": cells(df, predict)}


def main():
    nv = os.path.join(ROOT, "nvidia")
    out = []
    for d in DATASETS:
        if d.exists():
            mod = d.modality + (f" {d.variant}" if d.variant else "")
            out.append(entry("A100", d.model, mod, d.precision, os.path.join(nv, d.path), True))
    for model, prec, run in QUANT:
        p = os.path.join(nv, Z.format(run))
        if os.path.exists(p):
            out.append(entry("A100", model, "text", prec, p, True))
    for model, run in INLEN:                    # input-length sweeps: one entry per prompt length
        for n in (128, 512, 2048, 4096):
            p = os.path.join(nv, "results_zeus_inlen", f"run_text_only_{run}_fp16", f"in{n}", "profiling_metrics.csv")
            if os.path.exists(p):
                out.append(entry("A100", model, f"text, {n}-token prompts", "fp16", p, True))
    for model, mod, run, *prec in AURORA:
        p = os.path.join(ROOT, "intel", "results_aurora", f"run_{run}", "res_native", "profiling_metrics.csv")
        out.append(entry("Max 1550", model, mod, prec[0] if prec else "fp16", p, False))
    # MMLU (5-shot) per A100 checkpoint, and the layouts where vLLM 0.16 bitsandbytes gives correct output
    mmlu = {"Llama-3.1-8B": {"fp16": 68.4, "awq4bit": 66.4, "w8a8": 67.8},
            "Qwen2.5-14B": {"fp16": 79.9, "awq4bit": 78.9, "gptq8bit": 79.9, "bnb4bit": 79.3}}
    valid = {("Llama-3.1-8B", "bnb4bit"): [[1, 1], [4, 1]], ("Qwen2.5-14B", "bnb4bit"): [[1, 1]],
             ("Mistral-7B", "bnb4bit"): [[1, 1], [4, 1]]}
    for d in out:
        if d["hardware"] == "A100" and d["modality"] == "text":
            if d["precision"] in mmlu.get(d["model"], {}):
                d["mmlu"] = mmlu[d["model"]][d["precision"]]
            if (d["model"], d["precision"]) in valid:
                d["valid_layouts"] = valid[(d["model"], d["precision"])]
    meta = {"columns": ["max_tokens", "batch_size", "tp", "pp", "budget", "energy_J_per_request",
                        "latency_s", "power_W", "energylens_pred_J_per_request"],
            "datasets": out}
    os.makedirs(os.path.join(ROOT, "docs", "data"), exist_ok=True)
    path = os.path.join(ROOT, "docs", "data", "site.json")
    with open(path, "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    with open(path, "rb") as f, open(path[:-5] + ".js", "w") as g:   # same data as a script, for hosts
        g.write("window.SITE_DATA = " + f.read().decode() + ";\n")     # that do not serve JSON to fetch()
    print(f"{len(out)} datasets -> {path} ({os.path.getsize(path) / 1e6:.1f} MB) and site.js")


if __name__ == "__main__":
    main()
