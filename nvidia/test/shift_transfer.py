#!/usr/bin/env python3
"""Robustness to a change of data distribution, and transfer across models.

    python test/shift_transfer.py [--part A|B|all] [--seeds 42 7 123]
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize as opt, minimize_scalar

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
NVIDIA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(NVIDIA, "..", "adaptive_sampling"))
from compare_cross_model import (fit_unified, unified_predict, fit_maveriq, maveriq_predict,  # noqa: E402
                                 holdout_scenarios, get_ml_features, prepare_df, drop_invalid_rows)
from simulator import load, ranking_metrics  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402

OUT = os.path.join(NVIDIA, "results_figures", "shift_transfer")
os.makedirs(OUT, exist_ok=True)
SHAPE = [1, 2, 3, 4, 6, 7, 8, 9]          # beta, gamma1, gamma2, eps of prefill and decode
BOUNDS = [(1e-8, 1e6), (0.01, 3), (-2, 3), (-2, 3), (0, 100), (1e-8, 1e6), (0.01, 3), (-2, 3), (-2, 3),
          (0, 100), (-1e6, 1e6), (-1e6, 1e6)]


def mape(pred, y):
    return 100 * float(np.mean(np.abs(pred - y) / y))


def gb_fit(train):
    m = GradientBoostingRegressor(n_estimators=100, random_state=42)
    m.fit(get_ml_features(train), train["measured_energy"].values.astype(float))
    return lambda d: m.predict(get_ml_features(d))


def fit_all(train):
    """EnergyLens, MaverIQ (direct energy form) and Gradient Boosting on `train`."""
    th = fit_unified(train, restarts=10)
    mp = fit_maveriq(train, restarts=3)
    gb = gb_fit(train)
    return {"EnergyLens": lambda d: unified_predict(th, d), "MaverIQ": lambda d: maveriq_predict(mp, d),
            "Gradient Boosting": gb}


def score(fns, test):
    y = test["measured_energy"].to_numpy(float)
    out = {}
    for m, f in fns.items():
        p = np.asarray(f(test), float)
        r = ranking_metrics(test, p)
        out[m] = {"mape": mape(p, y), "pairwise": r["pairwise"], "regret": r["regret"]}
    return out


# ── A1: modality shift ───────────────────────────────────────────────────────────────
MODALITIES = {
    "Qwen2-VL-7B": {"text": "Qwen2-VL-7B", "image": "Qwen2-VL-7B (image)", "video": "Qwen2-VL-7B (video, 240p)",
                    "video-only": "Qwen2-VL-7B (video-only, 240p)"},
    "LLaVA-1.5-7B": {"text": "LLaVA-1.5-7B", "image": "LLaVA-1.5-7B (image)", "video": "LLaVA-1.5-7B (video, 240p)"},
}


def part_a1(seeds, n=50):
    rows = []
    for model, mods in MODALITIES.items():
        data = {m: load(k) for m, k in mods.items()}
        for src in mods:
            for seed in seeds:
                tr = data[src].sample(n, random_state=seed)
                fns = fit_all(tr)
                for tgt in mods:
                    if tgt == src:      # reference: held-out scenarios of the same dataset
                        pool, test, _, _ = holdout_scenarios(data[src], 0.3, seed)
                        tr2 = pool.sample(min(n, len(pool)), random_state=seed)
                        res = score(fit_all(tr2), test)
                    else:               # shifted: every row of the target
                        res = score(fns, data[tgt])
                    for meth, v in res.items():
                        rows.append({"model": model, "source": src, "target": tgt, "seed": seed,
                                     "method": meth, **v})
        print(f"[A1] {model} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "a1_modality.csv"), index=False)
    return d


# ── A2: conversation <-> code traces ────────────────────────────────────────────────
def load_trace(path):
    """Trace batches prepared like the grid rows (compare_cross_model's columns)."""
    raw = pd.read_csv(path)
    raw = raw[pd.to_numeric(raw["energy"], errors="coerce") > 0].reset_index(drop=True)
    raw = drop_invalid_rows(raw, path).reset_index(drop=True)
    df = prepare_df(raw, path)
    df["batch_id"], df["trace"] = raw.loc[df.index, "batch_id"].values, raw.loc[df.index, "trace"].values
    return df.reset_index(drop=True)


TRACE_MODELS = ["Mistral-7B-Instruct-v0.3", "Llama-3.1-8B-Instruct", "Qwen2.5-14B-Instruct",
                "Mistral-Small-24B-Instruct-2501", "Qwen2.5-32B-Instruct"]


def part_a2(seeds):
    rows = []
    for name in TRACE_MODELS:
        t = load_trace(os.path.join(NVIDIA, "results_zeus_trace", f"run_trace_{name}_fp16", "profiling_metrics.csv"))
        sets = {k: t[t["trace"] == k].reset_index(drop=True) for k in ("conv", "code")}
        for src, tgt in (("conv", "code"), ("code", "conv")):
            fns = fit_all(sets[src])
            for meth, v in score(fns, sets[tgt]).items():
                rows.append({"model": name, "source": src, "target": tgt, "kind": "shift", "method": meth, **v})
            # reference: same distribution, half of the batches to fit, the other half to test
            for seed in seeds:
                ids = np.random.RandomState(seed).permutation(sets[src]["batch_id"].unique())
                a = sets[src][sets[src]["batch_id"].isin(ids[: len(ids) // 2])]
                b = sets[src][sets[src]["batch_id"].isin(ids[len(ids) // 2:])]
                for meth, v in score(fit_all(a), b).items():
                    rows.append({"model": name, "source": src, "target": src, "kind": f"same{seed}",
                                 "method": meth, **v})
        print(f"[A2] {name} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "a2_traces.csv"), index=False)
    return d


# ── B: cross-model transfer ─────────────────────────────────────────────────────────
TEXT = ["Mamba-130M", "Falcon-Mamba-7B", "Nemotron-Nano-9B", "Mistral-7B", "Llama-3.1-8B", "Qwen2.5-14B",
        "Mistral-Small-24B", "Qwen2.5-32B", "Qwen1.5-MoE-2.7B", "Mixtral-8x7B", "Qwen2-VL-7B", "LLaVA-1.5-7B"]


def fit_scale(theta, tr):
    y = tr["measured_energy"].to_numpy(float)
    p = unified_predict(theta, tr)
    f = lambda ls: np.mean(np.abs(np.exp(ls) * p - y) / y)
    r = minimize_scalar(f, bounds=(-9, 9), method="bounded")
    return float(np.exp(r.x))


def fit_prior(tr, theta0, lam):
    """Eq. (unified-tp-pp) on the k target rows, started at the other models' median parameters
    (scale-corrected) and pulled towards their shape exponents."""
    y = tr["measured_energy"].to_numpy(float)
    s = fit_scale(theta0, tr)
    start = np.array(theta0, float)
    for j in (0, 5, 10, 11):
        start[j] *= s
    ref = np.array(theta0, float)

    def loss(p):
        return np.mean(np.abs(unified_predict(p, tr) - y) / y) + lam * np.sum((p[SHAPE] - ref[SHAPE]) ** 2)

    r = opt(loss, start, method="L-BFGS-B", bounds=BOUNDS, options={"maxiter": 5000, "ftol": 1e-12})
    return r.x


def part_b(seeds, ks=(5, 10, 25, 50)):
    data = {m: load(m) for m in TEXT}
    thetas = {m: fit_unified(data[m], restarts=8) for m in TEXT}
    print("[B] fitted all models", flush=True)
    pd.DataFrame(thetas).T.to_csv(os.path.join(OUT, "b_theta_per_model.csv"))
    rows = []
    for tgt in TEXT:
        others = np.array([thetas[m] for m in TEXT if m != tgt])
        theta0 = np.median(others, axis=0)
        for seed in seeds:
            pool, test, _, _ = holdout_scenarios(data[tgt], 0.3, seed)
            y = test["measured_energy"].to_numpy(float)
            base = {"target": tgt, "seed": seed}
            for k in ks:
                tr = pool.sample(min(k, len(pool)), random_state=seed)
                preds = {"scratch": unified_predict(fit_unified(tr, restarts=10), test),
                         "scale": fit_scale(theta0, tr) * unified_predict(theta0, test),
                         "Gradient Boosting": gb_fit(tr)(test)}
                for lam in (0.1, 1.0, 10.0):
                    preds[f"refit l={lam:g}"] = unified_predict(fit_prior(tr, theta0, lam), test)
                for meth, p in preds.items():
                    r = ranking_metrics(test, np.asarray(p, float))
                    rows.append({**base, "k": k, "method": meth, "mape": mape(np.asarray(p, float), y),
                                 "pairwise": r["pairwise"], "regret": r["regret"]})
            p0 = unified_predict(theta0, test)      # k = 0: the other models' median, unscaled
            r = ranking_metrics(test, p0)
            rows.append({**base, "k": 0, "method": "prior, no data", "mape": mape(p0, y),
                         "pairwise": r["pairwise"], "regret": r["regret"]})
        print(f"[B] {tgt} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "b_cross_model.csv"), index=False)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part", default="all", choices=["A", "B", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    if a.part in ("A", "all"):
        f = os.path.join(OUT, "a1_modality.csv")
        d = pd.read_csv(f) if os.path.exists(f) else part_a1(a.seeds)
        g = d.assign(shift=np.where(d.source == d.target, "same distribution", "shifted"))
        print("\n== A1 modality shift: MAPE (%) mean over source/target pairs, models and seeds")
        print(g.groupby(["shift", "method"])[["mape", "pairwise", "regret"]].mean().round(1))
        print(d[d.source != d.target].pivot_table(index=["model", "source"], columns=["target"], values="mape",
                                                  aggfunc="mean").round(1).to_string())
        d2 = part_a2(a.seeds)
        print("\n== A2 conversation <-> code traces")
        d2["shift"] = np.where(d2.kind == "shift", "shifted", "same distribution")
        print(d2.groupby(["shift", "method"])[["mape", "pairwise", "regret"]].mean().round(1))
    if a.part in ("B", "all"):
        d = part_b(a.seeds)
        print("\n== B cross-model transfer, mean over the 12 target models")
        print(d.pivot_table(index="method", columns="k", values="mape").round(1).to_string())
        print(d.pivot_table(index="method", columns="k", values="pairwise").round(1).to_string())
        print(d.pivot_table(index="method", columns="k", values="regret").round(2).to_string())


if __name__ == "__main__":
    main()
