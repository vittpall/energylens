#!/usr/bin/env python3
"""GP and two-stage baselines where extrapolation matters.

  8-GPU layouts: fit on 50 rows of layouts of at most 4 GPUs, score on held-out 8-GPU scenarios
  LOCO (MAPE): withhold each (TP, PP) layout in turn, fit on 50 rows of the others, MAPE on it

    python test/more_baselines_extrap.py
"""
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
from compare_cross_model import fit_unified, unified_predict  # noqa: E402
from phi_and_baselines import loglinear_fit, gblog_fit  # noqa: E402
from more_baselines import gp_fit, twostage_fit  # noqa: E402
from gpu8_extrapolation import MODELS, path, SCEN  # noqa: E402


def methods(train):
    th = fit_unified(train, restarts=10)
    return {"EnergyLens": lambda d: unified_predict(th, d), "log-linear": loglinear_fit(train),
            "GB, log features": gblog_fit(train), "GP": gp_fit(train),
            "two-stage (GB)": twostage_fit(train, "gb"), "two-stage (linear)": twostage_fit(train, "linear")}


def gpu8(task):
    m, seed = task
    d4, d8 = load_csv(path("results_zeus", m)), load_csv(path("results_zeus_8gpu", m))
    d8 = d8[d8.tp_degree * d8.pp_degree == 8].reset_index(drop=True)
    small = d4[d4.tp_degree * d4.pp_degree <= 4]
    rng = np.random.RandomState(seed)
    scen = sorted(set(map(tuple, d8[SCEN].to_numpy())))
    ts = set(map(tuple, np.array(scen)[rng.permutation(len(scen))[: max(1, int(0.3 * len(scen)))]]))
    test = d8[np.array([tuple(k) in ts for k in d8[SCEN].to_numpy()])]
    y = test["measured_energy"].to_numpy(float)
    rows = []
    for name, f in methods(small.sample(min(50, len(small)), random_state=seed)).items():
        p = np.asarray(f(test), float)
        rows.append({"model": m, "seed": seed, "method": name, "mape": 100 * np.mean(np.abs(p - y) / y), **ranking_metrics(test, p)})
    return rows


def loco(task):
    name, seed = task
    d = load_csv(PATHS[name])
    cfgs = sorted(set(zip(d.tp_degree, d.pp_degree)))
    rows = []
    if len(cfgs) < 3:
        return rows
    for c in cfgs:
        is_c = (d.tp_degree == c[0]) & (d.pp_degree == c[1])
        rest, held = d[~is_c], d[is_c]
        try:
            fns = methods(rest.sample(min(50, len(rest)), random_state=seed))
        except Exception:
            continue
        y = held["measured_energy"].to_numpy(float)
        for mname, f in fns.items():
            p = np.asarray(f(held), float)
            rows.append({"dataset": name, "seed": seed, "cfg": str(c), "method": mname, "mape": 100 * np.mean(np.abs(p - y) / y)})
    return rows


def main():
    out = os.path.join(NVIDIA, "results_figures", "gpu8")
    with ProcessPoolExecutor(max_workers=16) as ex:
        r8 = [x for r in ex.map(gpu8, [(m, s) for m in MODELS for s in (42, 7, 123)]) for x in r]
        pd.DataFrame(r8).to_csv(os.path.join(out, "more_baselines_gpu8.csv"), index=False)
        print("8-GPU layouts (mean of 4 models x 3 seeds)")
        print(pd.DataFrame(r8).groupby("method")[["mape", "pairwise", "top1", "regret"]].mean().round(1).to_string(), flush=True)
        rl = [x for r in ex.map(loco, [(n, s) for n in PATHS for s in (42,)]) for x in r]
    l = pd.DataFrame(rl)
    l.to_csv(os.path.join(out, "more_baselines_loco_mape.csv"), index=False)
    print("\nLOCO MAPE on the withheld layout (mean over layouts, then datasets; seed 42)")
    print(l.groupby(["method", "dataset"]).mape.mean().groupby("method").mean().round(1).to_string())


if __name__ == "__main__":
    main()
