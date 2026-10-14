"""Offline profiling simulator over complete Zeus grids.

A dataset's rows are split into held-out test scenarios and a candidate pool.
A strategy only sees the energy of pool rows it has measured, and every
measurement is charged like the real profiler: an engine reload whenever the
(TP, PP, token budget) changes, plus the batch's measured latency.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NVIDIA = os.path.join(HERE, "..", "nvidia")
sys.path.insert(0, os.path.join(NVIDIA, "test"))
from compare_cross_model import (  # noqa: E402
    prepare_df, drop_invalid_rows, fit_unified, unified_predict, holdout_scenarios,
)

ZEUS = os.path.join(NVIDIA, "results_zeus")
DATASETS = {
    "Mistral-7B":        "run_text_only_Mistral-7B-Instruct-v0.3_fp16",
    "Mistral-Small-24B": "run_text_only_Mistral-Small-24B-Instruct-2501_fp16",
    "Nemotron-Nano-9B":  "run_text_only_NVIDIA-Nemotron-Nano-9B-v2_fp16",
    "Qwen2.5-14B":       "run_text_only_Qwen2.5-14B-Instruct_fp16",
    "Falcon-Mamba-7B":   "run_text_only_falcon-mamba-7b_fp16",
    "Mamba-130M":        "run_text_only_mamba-130m-hf_fp16",
    "Qwen1.5-MoE-2.7B":  "run_text_only_Qwen1.5-MoE-A2.7B_fp16",
}
# Seconds per engine reload: median wall-clock start-up in the model's
# profiling log (engine creation -> "init engine ... took") + the profiler's
# 10 s pause between configurations.
RELOAD_S = {
    "Mistral-7B": 58.0, "Mistral-Small-24B": 73.0, "Nemotron-Nano-9B": 60.5,
    "Qwen2.5-14B": 69.0, "Falcon-Mamba-7B": 51.0, "Mamba-130M": 43.0,
    "Qwen1.5-MoE-2.7B": 62.5,
}
ENGINE = ["tp_degree", "pp_degree", "budget"]

# Every other dataset of the paper's registry (nvidia/viz/datasets.py), keyed by its
# label, for run.py --all. The 7 above keep their names (profiling_cost.py uses them).
# Reload seconds per model: median start-up in nvidia/logs (V1 engine init -> "init
# engine ... took", 21-401 engines per model) + 20 s. That log span starts after the
# engine is created, so it is shorter than the one RELOAD_S measures; +20 s (incl.
# the 10 s pause) reproduces RELOAD_S within 1.5 s for the models in both.
RELOAD_BY_MODEL = {
    "Llama-3.1-8B": 56.0, "Qwen2.5-32B": 95.0, "Mixtral-8x7B": 70.0, "Qwen2-VL-7B": 70.0,
    "LLaVA-1.5-7B": 60.0, "InternVL2-76B": 109.0,
    "Mistral-7B": 58.0, "Qwen2.5-14B": 69.0,   # bnb4bit variants: same start-up as fp16
}
PATHS = {name: os.path.join(ZEUS, run, "res_native", "profiling_metrics.csv")
         for name, run in DATASETS.items()}
sys.path.insert(0, NVIDIA)
from viz.datasets import DATASETS as _REGISTRY  # noqa: E402
for _d in _REGISTRY:
    _p = os.path.normpath(os.path.join(NVIDIA, _d.path))
    if _d.exists() and _p not in map(os.path.normpath, PATHS.values()) and _d.model in RELOAD_BY_MODEL:
        PATHS[_d.label] = _p
        RELOAD_S[_d.label] = RELOAD_BY_MODEL[_d.model]


def load(name):
    return load_csv(PATHS[name])


def load_csv(path):
    raw = pd.read_csv(path).dropna(subset=["latency", "power", "energy"]).reset_index(drop=True)
    raw = raw[pd.to_numeric(raw["energy"], errors="coerce") > 0].reset_index(drop=True)
    # drop invalid rows here, so the raw column copied below stays aligned with df
    raw = drop_invalid_rows(raw, path)
    df = prepare_df(raw, path)
    df["budget"] = raw["max_num_batched_tokens"].astype(int).values
    return df


class Simulator:
    def __init__(self, name, seed, test_frac=0.3):
        self.name = name
        self.pool, self.test, _, _ = holdout_scenarios(load(name), test_frac, seed)
        self.reload_s = RELOAD_S[name]
        self.engine = list(zip(*(self.pool[c].tolist() for c in ENGINE)))
        self.latency = self.pool["measured_latency"].to_numpy(float)
        self.measured, self.cost, self.reloads, self.loaded = [], 0.0, 0, None

    def available(self):
        taken = set(self.measured)
        return [i for i in range(len(self.pool)) if i not in taken]

    def measure(self, rows, limit=np.inf):
        """Measure rows grouped by engine: the loaded one first, then in the order given. Stops at `limit` seconds."""
        first = {}
        for i in rows:
            first.setdefault(self.engine[i], len(first))
        rows = sorted(rows, key=lambda i: (self.engine[i] != self.loaded, first[self.engine[i]]))
        for i in rows:
            step = self.latency[i] + (self.reload_s if self.engine[i] != self.loaded else 0.0)
            if self.cost + step > limit:
                return False
            if self.engine[i] != self.loaded:
                self.loaded, self.reloads = self.engine[i], self.reloads + 1
            self.cost += step
            self.measured.append(i)
        return True

    def observed(self):
        return self.pool.iloc[self.measured]


def ranking_metrics(test, pred):
    """Pairwise accuracy, Top-1 and regret of the per-configuration ranking in each test scenario."""
    t = test.assign(pred=pred)
    pair, top1, regret = [], [], []
    for _, g in t.groupby(["max_tokens", "batch_size"]):
        m = g.groupby(["tp_degree", "pp_degree"])[["measured_energy", "pred"]].mean()
        if len(m) < 2:
            continue
        y, p = m["measured_energy"].to_numpy(), m["pred"].to_numpy()
        dy, dp = y[:, None] - y[None, :], p[:, None] - p[None, :]
        valid = np.triu(np.abs(dy) > 1e-9, 1)
        correct = (dy * dp > 0) + 0.5 * (np.abs(dp) < 1e-9)
        pair.append(correct[valid].mean())
        best, pick = y.argmin(), p.argmin()
        top1.append(best == pick)
        regret.append((y[pick] - y[best]) / y[best] * 100)
    return {"pairwise": 100 * np.mean(pair), "top1": 100 * np.mean(top1), "regret": np.mean(regret)}


def evaluate(train, test, restarts=20):
    theta = fit_unified(train, restarts=restarts)
    pred = unified_predict(theta, test)
    y = test["measured_energy"].to_numpy(float)
    return {"mape": 100 * np.mean(np.abs(pred - y) / y), **ranking_metrics(test, pred)}
