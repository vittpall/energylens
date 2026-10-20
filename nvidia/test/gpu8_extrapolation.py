#!/usr/bin/env python3
"""Fit on layouts of at most 4 GPUs (50 random rows), predict the four 8-GPU layouts.

k extra rows from the 8-GPU layouts (scenarios not in the test set) are added to the fit;
k = 0 is pure extrapolation. Scored on the held-out 8-GPU scenarios (30% of them): MAPE,
pairwise accuracy and regret of the ranking of the four 8-GPU layouts.

    python test/gpu8_extrapolation.py [--seeds 42 7 123] [--ks 0 8 24]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NVIDIA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(NVIDIA, "..", "adaptive_sampling"))
from simulator import load_csv, ranking_metrics  # noqa: E402
from shift_transfer import fit_all  # noqa: E402

MODELS = ["Llama-3.1-8B-Instruct", "Qwen2.5-14B-Instruct", "Mistral-7B-Instruct-v0.3", "Qwen2.5-32B-Instruct"]
SCEN = ["max_tokens", "batch_size"]


def path(root, m):
    return os.path.join(NVIDIA, root, f"run_text_only_{m}_fp16", "res_native", "profiling_metrics.csv")


def run_one(task):
    m, seed, ks = task
    d4, d8 = load_csv(path("results_zeus", m)), load_csv(path("results_zeus_8gpu", m))
    d8 = d8[d8.tp_degree * d8.pp_degree == 8].reset_index(drop=True)
    small = d4[d4.tp_degree * d4.pp_degree <= 4]
    rng = np.random.RandomState(seed)
    scen = sorted(set(map(tuple, d8[SCEN].to_numpy())))
    test_s = set(map(tuple, np.array(scen)[rng.permutation(len(scen))[: max(1, int(0.3 * len(scen)))]]))
    is_test = np.array([tuple(k) in test_s for k in d8[SCEN].to_numpy()])
    test, cal = d8[is_test], d8[~is_test]
    base = small.sample(min(50, len(small)), random_state=seed)
    rows = []
    for k in ks:
        train = base if k == 0 else pd.concat([base, cal.sample(min(k, len(cal)), random_state=seed)])
        fns = fit_all(train)
        y = test["measured_energy"].to_numpy(float)
        for name, f in fns.items():
            p = np.asarray(f(test), float)
            rows.append({"model": m, "seed": seed, "k": k, "method": name,
                         "mape": 100 * np.mean(np.abs(p - y) / y), **ranking_metrics(test, p)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 8, 24])
    ap.add_argument("--out", default=os.path.join(NVIDIA, "results_figures", "gpu8"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = [(m, s, a.ks) for m in MODELS for s in a.seeds]
    rows = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for r in ex.map(run_one, tasks):
            rows += r
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(a.out, "gpu8_rows.csv"), index=False)
    pd.set_option("display.width", 200)
    cols = ["mape", "pairwise", "top1", "regret"]
    print(r.groupby(["model", "k", "method"])[cols].mean().round(1).to_string())
    print(r.groupby(["k", "method"])[cols].mean().round(1).to_string())


if __name__ == "__main__":
    main()
