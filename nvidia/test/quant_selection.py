#!/usr/bin/env python3
"""Choosing the precision and the layout together under an accuracy limit (A100).

Candidates per held-out scenario (output limit, batch size): every checkpoint of the model
(FP16 and its quantized formats) x every measured layout and token budget. A checkpoint is
allowed when its MMLU drop from FP16 is at most delta (MMLU measured once per checkpoint, as
MaverIQ's accuracy intent uses known accuracies). Each method fits every checkpoint on 50
random rows of its training scenarios and picks the lowest predicted energy among the allowed
candidates; it is charged the measured energy of its pick.

    python test/quant_selection.py [--seeds 42 7 123] [--deltas 0 0.5 1 2]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NVIDIA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(NVIDIA, "..", "adaptive_sampling"))
from simulator import load_csv  # noqa: E402
from shift_transfer import fit_all  # noqa: E402

Z = os.path.join(NVIDIA, "results_zeus", "run_text_only_{}", "res_native", "profiling_metrics.csv")
# checkpoint -> (run, MMLU 5-shot, layouts with correct output or None for all)
MODELS = {
    "Llama-3.1-8B": {
        "FP16": ("Llama-3.1-8B-Instruct_fp16", 68.4, None),
        "AWQ 4-bit": ("Meta-Llama-3.1-8B-Instruct-AWQ-INT4_awq4bit", 66.4, None),
        "W8A8": ("Meta-Llama-3.1-8B-Instruct-quantized.w8a8_w8a8", 67.8, None),
    },
    "Qwen2.5-14B": {
        "FP16": ("Qwen2.5-14B-Instruct_fp16", 79.9, None),
        "AWQ 4-bit": ("Qwen2.5-14B-Instruct-AWQ_awq4bit", 78.9, None),
        "GPTQ 8-bit": ("Qwen2.5-14B-Instruct-GPTQ-Int8_gptq8bit", 79.9, None),
        "bnb 4-bit": ("Qwen2.5-14B-Instruct_bnb4bit", 79.3, {(1, 1)}),   # vLLM 0.16: garbage elsewhere
    },
}
SCEN = ["max_tokens", "batch_size"]
CELL = SCEN + ["tp_degree", "pp_degree", "budget"]


def load(run, layouts):
    d = load_csv(Z.format(run))
    if layouts:
        d = d[[(t, p) in layouts for t, p in zip(d.tp_degree, d.pp_degree)]].reset_index(drop=True)
    return d


def run(model, ckpts, seed, deltas):
    data = {c: load(r, lay) for c, (r, _, lay) in ckpts.items()}
    mmlu = {c: v[1] for c, v in ckpts.items()}
    scen = set.intersection(*(set(map(tuple, d[SCEN].drop_duplicates().to_numpy())) for d in data.values()))
    scen = sorted(scen)
    rng = np.random.RandomState(seed)
    test = set(map(tuple, np.array(scen)[rng.permutation(len(scen))[: max(1, int(0.3 * len(scen)))]]))
    cells = []
    for c, d in data.items():
        key = list(map(tuple, d[SCEN].to_numpy()))
        is_test = np.array([k in test for k in key])
        train = d[~is_test]
        fns = fit_all(train.sample(min(50, len(train)), random_state=seed))
        te = d[is_test].copy()
        for m, f in fns.items():
            te[m] = np.asarray(f(te), float)
        g = te.groupby(CELL).agg(y=("measured_energy", "mean"),
                                 **{m: (m, "mean") for m in fns}).reset_index()
        g["ckpt"] = c
        cells.append(g)
    cells = pd.concat(cells, ignore_index=True)
    methods = [m for m in cells.columns if m not in CELL + ["y", "ckpt"]]
    rows = []
    for delta in deltas:
        allowed = [c for c in ckpts if mmlu["FP16"] - mmlu[c] <= delta + 1e-9]
        for s, g in cells.groupby(SCEN):
            a = g[g.ckpt.isin(allowed)]
            best = a.y.min()
            fp16 = g[g.ckpt == "FP16"]
            out = {"oracle": a.y.min(), "best FP16 (measured)": fp16.y.min()}
            for m in methods:
                out[m] = a.y.loc[a[m].idxmin()]
                out[f"{m}, FP16 only"] = fp16.y.loc[fp16[m].idxmin()]
            for pol, e in out.items():
                rows.append({"model": model, "seed": seed, "delta": delta, "scenario": s, "policy": pol,
                             "energy": e, "regret": 100 * (e / best - 1), "top1": float(abs(e - best) < 1e-9),
                             "saving_vs_fp16": 100 * (1 - e / fp16.y.min())})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--deltas", type=float, nargs="+", default=[0, 0.5, 1, 2])
    ap.add_argument("--out", default=os.path.join(NVIDIA, "results_figures", "quant_selection"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for model, ckpts in MODELS.items():
        for seed in a.seeds:
            rows += run(model, ckpts, seed, a.deltas)
            print(f"done {model} seed {seed}", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(a.out, "quant_selection_rows.csv"), index=False)
    s = r.groupby(["delta", "policy"]).agg(regret=("regret", "mean"), top1=("top1", "mean"),
                                           saving=("saving_vs_fp16", "mean")).reset_index()
    s["top1"] *= 100
    s.to_csv(os.path.join(a.out, "quant_selection_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(s.round(2).to_string(index=False))
    print("\nper model (delta = 1):")
    print(r[r.delta == 1].groupby(["model", "policy"]).agg(regret=("regret", "mean"), top1=("top1", "mean"),
                                                            saving=("saving_vs_fp16", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()
