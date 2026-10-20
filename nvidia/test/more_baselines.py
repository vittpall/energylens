#!/usr/bin/env python3
"""Two more learned baselines, same protocol as test/phi_and_baselines.py.

  GP                 Gaussian process (ARD RBF + white noise) on standardized log features,
                     log energy target
  two-stage (GB)     Gradient Boosting latency model x Gradient Boosting power model on log
                     features, energy = latency x power / batch size (it is exact in the data)
  two-stage (linear) the same with ordinary least squares on log features

    python test/more_baselines.py [--seeds 42 7 123] [--workers 16]
"""
import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NVIDIA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(NVIDIA, "..", "adaptive_sampling"))
from simulator import PATHS, load_csv  # noqa: E402
from compare_cross_model import holdout_scenarios  # noqa: E402
from phi_and_baselines import log_features, score  # noqa: E402


def feats(d):
    return np.column_stack([log_features(d), d["tp_degree"].values, d["pp_degree"].values])


def gp_fit(train):
    X = feats(train)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    k = ConstantKernel(1.0) * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(1e-2)
    gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit((X - mu) / sd, np.log(train["measured_energy"].values.astype(float)))
    return lambda d: np.exp(gp.predict((feats(d) - mu) / sd))


def reg(kind):
    if kind == "gb":
        return GradientBoostingRegressor(n_estimators=100, random_state=42)
    from sklearn.linear_model import LinearRegression
    return LinearRegression()


def twostage_fit(train, kind):
    X = log_features(train) if kind == "linear" else feats(train)
    XX = (lambda d: log_features(d)) if kind == "linear" else feats
    lat = reg(kind).fit(X, np.log(train["measured_latency"].values.astype(float)))
    pw = reg(kind).fit(X, np.log(train["gpu_power_w"].values.astype(float)))
    return lambda d: np.exp(lat.predict(XX(d)) + pw.predict(XX(d))) / d["batch_size"].values.astype(float)


def run_one(task):
    name, seed = task
    out = []
    try:
        d = load_csv(PATHS[name])
        pool, test, _, _ = holdout_scenarios(d, 0.3, seed)
        train = pool.sample(min(50, len(pool)), random_state=seed)
        for m, f in {"GP": gp_fit(train), "two-stage (GB)": twostage_fit(train, "gb"),
                     "two-stage (linear)": twostage_fit(train, "linear")}.items():
            out.append({"dataset": name, "seed": seed, "method": m, **score(f(test), test)})
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
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(run_one, [(n, s) for n in PATHS for s in a.seeds]):
            rows += r
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(a.out, "more_baseline_scores.csv"), index=False)
    print(s.groupby(["method", "dataset"])[["mape", "pairwise", "top1", "regret"]].mean().groupby("method").mean().round(2).to_string())


if __name__ == "__main__":
    main()
