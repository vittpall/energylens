#!/usr/bin/env python3
"""Fitted-parameter stability of EnergyLens, and two stronger learned baselines.

Same protocol as test/ablation.py (29 A100 datasets, three split seeds, 50 random rows
of the training scenarios, scored on held-out scenarios).

  phi      the 12 fitted parameters of every (dataset, seed), plus a fit on all rows
  log-linear         ordinary least squares of log E on log T_in, log T_out, log b_eff,
                     log TP, log PP (a generic power law, no template structure)
  GB, log features   Gradient Boosting (100 trees) on the same log features, plus raw TP
                     and PP
  EnergyLens         the template, for reference on the same splits
  Gradient Boosting  the paper's baseline (raw features), for reference

    python test/phi_and_baselines.py [--seeds 42 7 123] [--workers 16]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NVIDIA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(NVIDIA, "..", "adaptive_sampling"))
from simulator import PATHS, load_csv, ranking_metrics  # noqa: E402
from compare_cross_model import holdout_scenarios, effective_batch, fit_unified, unified_predict  # noqa: E402
from shift_transfer import gb_fit  # noqa: E402

NAMES = ["alpha_p", "beta_p", "gamma1_p", "gamma2_p", "eps_p",
         "alpha_d", "beta_d", "gamma1_d", "gamma2_d", "eps_d", "delta1", "delta2"]


def log_features(d):
    return np.column_stack([np.log(d["total_input"].values.astype(float) + 1),
                            np.log(d["decode_tokens"].values.astype(float) + 1),
                            np.log(effective_batch(d)),
                            np.log(d["tp_degree"].values.astype(float)),
                            np.log(d["pp_degree"].values.astype(float))])


def loglinear_fit(train):
    X = np.column_stack([np.ones(len(train)), log_features(train)])
    w = np.linalg.lstsq(X, np.log(train["measured_energy"].values.astype(float)), rcond=None)[0]
    return lambda d: np.exp(np.column_stack([np.ones(len(d)), log_features(d)]) @ w)


def gblog_fit(train):
    def X(d):
        return np.column_stack([log_features(d), d["tp_degree"].values, d["pp_degree"].values])
    m = GradientBoostingRegressor(n_estimators=100, random_state=42)
    m.fit(X(train), np.log(train["measured_energy"].values.astype(float)))
    return lambda d: np.exp(m.predict(X(d)))


def score(pred, test):
    y = test["measured_energy"].values.astype(float)
    p = np.asarray(pred, float)
    return {"mape": 100 * np.mean(np.abs(p - y) / y), **ranking_metrics(test, p)}


def run_one(task):
    name, seed = task
    out = {"scores": [], "phi": []}
    try:
        d = load_csv(PATHS[name])
        pool, test, _, _ = holdout_scenarios(d, 0.3, seed)
        train = pool.sample(min(50, len(pool)), random_state=seed)
        th = fit_unified(train, restarts=20)
        out["phi"].append({"dataset": name, "seed": seed, "fit": "50 rows", **dict(zip(NAMES, th))})
        if seed == 42:
            th_all = fit_unified(d, restarts=20)
            out["phi"].append({"dataset": name, "seed": -1, "fit": "all rows", **dict(zip(NAMES, th_all))})
        methods = {"EnergyLens": lambda t: unified_predict(th, t), "log-linear": loglinear_fit(train),
                   "GB, log features": gblog_fit(train), "Gradient Boosting": gb_fit(train)}
        for m, f in methods.items():
            out["scores"].append({"dataset": name, "seed": seed, "method": m, **score(f(test), test)})
    except Exception as e:
        print(f"[skip] {name} {seed}: {type(e).__name__}: {e}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(NVIDIA, "results_figures", "phi_baselines"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    scores, phi = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(run_one, [(n, s) for n in PATHS for s in a.seeds]):
            scores += r["scores"]
            phi += r["phi"]
            if r["scores"]:
                print(f"done {r['scores'][0]['dataset']} seed {r['scores'][0]['seed']}", flush=True)
    s, p = pd.DataFrame(scores), pd.DataFrame(phi)
    s.to_csv(os.path.join(a.out, "baseline_scores.csv"), index=False)
    p.to_csv(os.path.join(a.out, "phi.csv"), index=False)
    cols = ["mape", "pairwise", "top1", "regret"]
    print(s.groupby(["method", "dataset"])[cols].mean().groupby("method").mean().round(2).to_string())


if __name__ == "__main__":
    main()
