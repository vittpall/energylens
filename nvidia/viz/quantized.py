#!/usr/bin/env python3
"""Quantized checkpoints against FP16 on A100.

  (a, b) energy per output token versus (TP, PP) layout, one line per precision,
         averaged over the grid cells every precision of that model measured
  (c)    MMLU (5-shot) versus energy per output token, one point per checkpoint
         (drawn once results_figures/mmlu/ holds the lm-eval results)

    python viz/quantized.py [--out results_figures/quantized]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from style import plt, save

HERE = os.path.dirname(os.path.abspath(__file__))
NVIDIA = os.path.dirname(HERE)
Z = "results_zeus/run_text_only_{}/res_native/profiling_metrics.csv"
MODELS = {
    "Llama-3.1-8B": {"FP16": "Llama-3.1-8B-Instruct_fp16",
                     "AWQ 4-bit": "Meta-Llama-3.1-8B-Instruct-AWQ-INT4_awq4bit",
                     "W8A8": "Meta-Llama-3.1-8B-Instruct-quantized.w8a8_w8a8"},
    "Qwen2.5-14B": {"FP16": "Qwen2.5-14B-Instruct_fp16",
                    "AWQ 4-bit": "Qwen2.5-14B-Instruct-AWQ_awq4bit",
                    "GPTQ 8-bit": "Qwen2.5-14B-Instruct-GPTQ-Int8_gptq8bit",
                    "bnb 4-bit": "Qwen2.5-14B-Instruct_bnb4bit"},
}
MMLU_TAG = {("Llama-3.1-8B", "FP16"): "llama31-8b_fp16", ("Llama-3.1-8B", "AWQ 4-bit"): "llama31-8b_awq4bit",
            ("Llama-3.1-8B", "W8A8"): "llama31-8b_w8a8", ("Qwen2.5-14B", "FP16"): "qwen25-14b_fp16",
            ("Qwen2.5-14B", "bnb 4-bit"): "qwen25-14b_bnb4bit", ("Qwen2.5-14B", "AWQ 4-bit"): "qwen25-14b_awq4bit",
            ("Qwen2.5-14B", "GPTQ 8-bit"): "qwen25-14b_gptq8bit"}
STYLE = {"FP16": ("#333333", "o", "-"), "AWQ 4-bit": ("#1f77b4", "s", "--"), "W8A8": ("#2ca02c", "^", "-."),
         "GPTQ 8-bit": ("#9467bd", "D", "-."), "bnb 4-bit": ("#d62728", "v", ":")}
KEY = ["tensor_parallel_size", "pipeline_parallel_size", "max_tokens", "batch_size", "max_num_batched_tokens",
       "input_length"]
LAYOUTS = [(1, 1), (2, 1), (1, 2), (4, 1), (2, 2), (1, 4)]


def load(run):
    d = pd.read_csv(os.path.join(NVIDIA, Z.format(run)))
    d = d[pd.to_numeric(d.energy, errors="coerce") > 0]
    return d.assign(ept=d.energy * d.batch_size / d.generated_tokens)


def mmlu(tag):
    f = sorted(glob.glob(os.path.join(NVIDIA, "results_figures", "mmlu", tag, "*", "results_*.json")))
    if not f:
        return None
    r = json.load(open(f[-1]))["results"]["mmlu"]
    return 100 * r.get("acc,none", np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(NVIDIA, "results_figures", "quantized"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    rows = []
    for ax, (model, runs) in zip(axs[:2], MODELS.items()):
        data = {p: load(r) for p, r in runs.items()}
        cells = set.intersection(*[set(map(tuple, d[KEY].drop_duplicates().to_numpy())) for d in data.values()])
        for p, d in data.items():
            d = d[d[KEY].apply(tuple, axis=1).isin(cells)]
            m = d.groupby(["tensor_parallel_size", "pipeline_parallel_size"]).ept.mean()
            ys = [m.get(l, np.nan) for l in LAYOUTS]
            col, mk, ls = STYLE[p]
            ax.plot(range(len(LAYOUTS)), ys, color=col, marker=mk, ls=ls, ms=8, mec="white", lw=2, label=p)
            rows += [{"model": model, "precision": p, "tp": l[0], "pp": l[1], "energy_per_token_j": y}
                     for l, y in zip(LAYOUTS, ys)]
            rows.append({"model": model, "precision": p, "tp": "all", "pp": "all",
                         "energy_per_token_j": d.ept.mean(), "mmlu": mmlu(MMLU_TAG[(model, p)])})
        ax.set_xticks(range(len(LAYOUTS)), [f"{t},{q}" for t, q in LAYOUTS])
        ax.set_xlabel("Layout (TP, PP)")
        ax.set_ylabel("Energy per output token (J)")
        ax.set_yscale("log")
        ax.set_title(f"({'ab'[list(MODELS).index(model)]}) {model}")
        ax.legend(fontsize=12)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    r = pd.DataFrame(rows)
    c = axs[2]
    pts = r[(r.tp == "all") & r.mmlu.notna()]
    for model, mk in (("Llama-3.1-8B", "o"), ("Qwen2.5-14B", "s")):
        for x in pts[pts.model == model].itertuples():
            c.scatter(x.energy_per_token_j, x.mmlu, color=STYLE[x.precision][0], marker=mk, s=90,
                      edgecolor="white", zorder=3)
            c.annotate(f"{model.split('-')[0]} {x.precision}", (x.energy_per_token_j, x.mmlu),
                       textcoords="offset points", xytext=(6, 4), fontsize=11)
    if pts.empty:
        c.text(0.5, 0.5, "MMLU results pending", ha="center", va="center", transform=c.transAxes, fontsize=14)
    c.set_xlabel("Energy per output token (J), mean over the grid")
    c.set_ylabel("MMLU accuracy, 5-shot (%)")
    c.set_title("(c) Accuracy versus energy")
    c.grid(alpha=0.25)
    c.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig._laid_out = True
    save(fig, a.out, "quantized")
    r.to_csv(os.path.join(a.out, "quantized.csv"), index=False)
    print(r[r.tp == "all"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
