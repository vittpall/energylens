#!/usr/bin/env python3
"""Accuracy and selection quality versus the number of training rows (box-plot data).

For each A100 dataset, seed and n, EnergyLens, MaverIQ, Gradient Boosting and the further
baselines (log-linear, Gradient Boosting on log features, GP, two-stage latency x power) are fitted on n
random rows of the training scenarios and scored on the held-out 30% (MAPE, pairwise
accuracy, Top-1, regret). viz/sample_efficiency.py draws the box plots.

    python test/sample_efficiency.py [--n 10 25 50 100 200 400] [--seeds 42 7 123 1 2]
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
from simulator import PATHS, load_csv, ranking_metrics  # noqa: E402
from compare_cross_model import holdout_scenarios  # noqa: E402
from shift_transfer import fit_all  # noqa: E402
from phi_and_baselines import loglinear_fit, gblog_fit  # noqa: E402
from more_baselines import gp_fit, twostage_fit  # noqa: E402
from compare_cross_model import get_ml_features  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def sk_fit(train, est):
    """The paper's Linear Regression / Random Forest baselines: the EnergyLens input features,
    standardized, raw energy target (compare_cross_model.fit_predict_ml)."""
    sc = StandardScaler().fit(get_ml_features(train))
    est.fit(sc.transform(get_ml_features(train)), train["measured_energy"].values.astype(float))
    return lambda d: est.predict(sc.transform(get_ml_features(d)))


def run(task):
    name, seed, ns, only = task
    d = load_csv(PATHS[name])
    pool, test, _, _ = holdout_scenarios(d, 0.3, seed)
    y = test.measured_energy.to_numpy(float)
    rows = []
    for n in ns:
        if n > len(pool):
            continue
        try:
            tr = pool.sample(n, random_state=seed)
            fns = {"Linear Regression": sk_fit(tr, LinearRegression()),
                   "Random Forest": sk_fit(tr, RandomForestRegressor(n_estimators=100, random_state=42))}
            if not only:
                fns.update({**fit_all(tr), "log-linear": loglinear_fit(tr), "GB, log features": gblog_fit(tr),
                            "GP": gp_fit(tr), "two-stage (linear)": twostage_fit(tr, "linear"),
                            "two-stage (GB)": twostage_fit(tr, "gb")})
        except Exception as e:
            print(f"[skip] {name} {seed} {n}: {e}", flush=True)
            continue
        for m, f in fns.items():
            p = np.asarray(f(test), float)
            rows.append({"dataset": name, "seed": seed, "n": n, "method": m,
                         "mape": 100 * np.mean(np.abs(p - y) / y), **ranking_metrics(test, p)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, nargs="+", default=[10, 25, 50, 100, 200, 400])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123, 1, 2])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--add-sklearn-only", action="store_true",
                    help="only fit Linear Regression and Random Forest and append them to the csv")
    a = ap.parse_args()
    out = os.path.join(NVIDIA, "results_figures", "sample_efficiency")
    os.makedirs(out, exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        r = pd.DataFrame([x for rr in ex.map(run, [(n, s, a.n, a.add_sklearn_only) for n in PATHS for s in a.seeds])
                          for x in rr])
    path = os.path.join(out, "sample_efficiency.csv")
    if a.add_sklearn_only and os.path.exists(path):
        old = pd.read_csv(path)
        r = pd.concat([old[~old.method.isin(r.method.unique())], r], ignore_index=True)
    r.to_csv(path, index=False)
    print(r.groupby(["method", "n"])[["mape", "pairwise", "top1", "regret"]].median().round(2).to_string())


if __name__ == "__main__":
    main()
