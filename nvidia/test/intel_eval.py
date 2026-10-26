#!/usr/bin/env python3
"""EnergyLens on Intel Data Center GPU Max 1550, against the A100 results.

    python test/intel_eval.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
NVIDIA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(NVIDIA, "..", "adaptive_sampling"))
from compare_cross_model import fit_unified, unified_predict, holdout_scenarios  # noqa: E402
from simulator import load_csv, ranking_metrics  # noqa: E402
from shift_transfer import fit_all as _fit_all, score, mape, fit_scale, fit_prior, gb_fit  # noqa: E402

OUT = os.path.join(NVIDIA, "results_figures", "intel")
A = "../intel/results_intel/run_{}/res_native/profiling_metrics.csv"
Z = "results_zeus/run_{}/res_native/profiling_metrics.csv"
ZM = "results_zeus_maxgen/run_{}/res_native/profiling_metrics.csv"
# short name -> (Intel GPU CSV, A100 CSV)
RUNS = {
    "Mistral-7B": (A.format("text_only_Mistral-7B-Instruct-v0.3_fp16"), Z.format("text_only_Mistral-7B-Instruct-v0.3_fp16")),
    "Llama-3.1-8B": (A.format("text_only_Llama-3.1-8B-Instruct_fp16"), Z.format("text_only_Llama-3.1-8B-Instruct_fp16")),
    "Qwen2.5-14B": (A.format("text_only_Qwen2.5-14B-Instruct_fp16"), Z.format("text_only_Qwen2.5-14B-Instruct_fp16")),
    "Qwen2-VL-7B (image)": (A.format("text_image_Qwen2-VL-7B-Instruct_fp16"),
                            ZM.format("text_image_Qwen2-VL-7B-Instruct_fp16")),
    "Qwen1.5-MoE-A2.7B": (A.format("text_only_Qwen1.5-MoE-A2.7B_fp16"), Z.format("text_only_Qwen1.5-MoE-A2.7B_fp16")),
}
SEEDS = (42, 7, 123)
TDP = 600.0     # W per Max 1550 card (two tiles)
NAMES = ["alpha_p", "beta_p", "gamma1_p", "gamma2_p", "eps_p", "alpha_d", "beta_d", "gamma1_d", "gamma2_d",
         "eps_d", "delta1", "delta2"]


def odd_tiles(d):
    """Layouts that leave one tile of their last card idle: the card counter still charges it."""
    return ((d["tp_degree"] * d["pp_degree"]) % 2 == 1).to_numpy()


def idle_share(d):
    """Idle tiles per occupied tile: 1/n for a layout of n tiles with one idle tile, else 0."""
    n = (d["tp_degree"] * d["pp_degree"]).to_numpy()
    return odd_tiles(d) / n


def fit_card(train):
    """EnergyLens with card-level accounting: Eq. (unified-tp-pp) on tile counts, times
    (1 + kappa / n) on the layouts of n tiles that leave one tile idle (kappa: the idle tile's
    power relative to a working one); kappa and theta fitted alternately, 3 rounds."""
    y = train["measured_energy"].to_numpy(float)
    o = idle_share(train)
    kappa, th = 0.0, None
    for _ in range(3):
        th = fit_unified(train.assign(measured_energy=y / (1 + kappa * o)), restarts=10)
        p = unified_predict(th, train)
        m = o > 0
        if m.any() and (~m).any():
            kappa = max(float(np.median((y[m] / p[m] - 1) / o[m])), 0.0)
    return lambda d: unified_predict(th, d) * (1 + kappa * idle_share(d))


def fit_all(train):
    fns = _fit_all(train)
    fns["EnergyLens + card term"] = fit_card(train)
    return fns


def load(rel):
    return load_csv(os.path.join(NVIDIA, rel))


def accuracy(data):
    rows = []
    for name, d in data.items():
        for seed in SEEDS:
            pool, test, _, _ = holdout_scenarios(d, 0.3, seed)
            tr = pool.sample(min(50, len(pool)), random_state=seed)
            for meth, v in score(fit_all(tr), test).items():
                rows.append({"dataset": name, "seed": seed, "method": meth, **v})
    return pd.DataFrame(rows)


def loco(data):
    rows = []
    for name, d in data.items():
        pred = {m: np.zeros(len(d)) for m in ("EnergyLens", "EnergyLens + card term", "MaverIQ", "Gradient Boosting")}
        for (tp, pp), idx in d.groupby(["tp_degree", "pp_degree"]).groups.items():
            out = d.index.isin(idx)
            fns = fit_all(d[~out])
            for m, f in fns.items():
                pred[m][out] = np.asarray(f(d[out]), float)
        y = d["measured_energy"].to_numpy(float)
        for m, p in pred.items():
            rows.append({"dataset": name, "method": m, "mape": mape(p, y), **ranking_metrics(d, p)})
    return pd.DataFrame(rows)


def optimum(data):
    rows = []
    for name, d in data.items():
        g = d.groupby(["max_tokens", "batch_size", "tp_degree", "pp_degree"])[
            ["measured_latency", "measured_energy"]].mean().reset_index()
        for _, s in g.groupby(["max_tokens", "batch_size"]):
            if len(s) < 2:
                continue
            fast, cheap = s.loc[s.measured_latency.idxmin()], s.loc[s.measured_energy.idxmin()]
            rows.append({"dataset": name, "differ": (fast.tp_degree, fast.pp_degree) != (cheap.tp_degree, cheap.pp_degree),
                         "penalty": 100 * (fast.measured_energy / cheap.measured_energy - 1),
                         "fast": f"({int(fast.tp_degree)},{int(fast.pp_degree)})",
                         "cheap": f"({int(cheap.tp_degree)},{int(cheap.pp_degree)})"})
    return pd.DataFrame(rows)


def transfer(data, a100, ks=(5, 10, 25)):
    rows, thetas = [], []
    for name, d in data.items():
        th_a = fit_unified(a100[name], restarts=10)
        th_i = fit_unified(d, restarts=10)
        thetas += [{"dataset": name, "gpu": "A100", **dict(zip(NAMES, th_a))},
                   {"dataset": name, "gpu": "Max 1550", **dict(zip(NAMES, th_i))}]
        for seed in SEEDS:
            pool, test, _, _ = holdout_scenarios(d, 0.3, seed)
            y = test["measured_energy"].to_numpy(float)
            p0 = unified_predict(th_a, test)
            rows.append({"dataset": name, "seed": seed, "k": 0, "method": "A100 fit, as is",
                         "mape": mape(p0, y), **ranking_metrics(test, p0)})
            for k in ks:
                tr = pool.sample(k, random_state=seed)
                preds = {"A100 fit, rescaled": fit_scale(th_a, tr) * p0,
                         "A100 fit as prior": unified_predict(fit_prior(tr, th_a, 0.1), test),
                         "from scratch": unified_predict(fit_unified(tr, restarts=10), test),
                         "Gradient Boosting": gb_fit(tr)(test)}
                for m, p in preds.items():
                    p = np.asarray(p, float)
                    rows.append({"dataset": name, "seed": seed, "k": k, "method": m, "mape": mape(p, y),
                                 **ranking_metrics(test, p)})
        print(f"[transfer] {name} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(thetas)


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 200)
    data = {n: load(a) for n, (a, _) in RUNS.items()}
    a100 = {n: load(z) for n, (_, z) in RUNS.items()}
    for n, d in data.items():
        print(f"{n}: {len(d)} rows, {d.groupby(['tp_degree', 'pp_degree']).ngroups} layouts")

    acc = accuracy(data)
    acc.to_csv(os.path.join(OUT, "accuracy.csv"), index=False)
    print("\n== 50-row fits, held-out scenarios (mean over seeds)")
    print(acc.groupby(["dataset", "method"])[["mape", "pairwise", "regret"]].mean().round(2).unstack("method").to_string())
    print(acc.groupby("method")[["mape", "pairwise", "regret"]].mean().round(2))

    lo = loco(data)
    lo.to_csv(os.path.join(OUT, "loco.csv"), index=False)
    print("\n== leave one layout out")
    print(lo.pivot_table(index="dataset", columns="method", values=["pairwise", "regret"]).round(1).to_string())
    print(lo.groupby("method")[["mape", "pairwise", "regret"]].mean().round(2))

    op = optimum(data)
    op.to_csv(os.path.join(OUT, "optimum.csv"), index=False)
    print("\n== latency- vs energy-optimal layout")
    print(op.groupby("dataset").agg(scenarios=("differ", "size"), differ=("differ", "mean"),
                                    penalty=("penalty", "mean")).round(3).to_string())
    print("all:", len(op), "scenarios, differ", round(100 * op.differ.mean(), 1), "%, penalty",
          round(op.penalty.mean(), 1), "%")
    print(op.groupby("dataset")[["fast", "cheap"]].agg(lambda s: s.value_counts().index[0]).to_string())

    tr, th = transfer(data, a100)
    tr.to_csv(os.path.join(OUT, "transfer.csv"), index=False)
    th.to_csv(os.path.join(OUT, "theta.csv"), index=False)
    print("\n== A100 -> Intel GPU transfer: MAPE / pairwise / regret (mean over datasets and seeds)")
    for v in ("mape", "pairwise", "regret"):
        print(tr.pivot_table(index="method", columns="k", values=v).round(2).to_string())
    print("\n== fitted exponents")
    print(th.set_index(["dataset", "gpu"])[["beta_p", "gamma1_p", "gamma2_p", "beta_d", "gamma1_d", "gamma2_d",
                                           "delta1", "delta2"]].round(2).to_string())


if __name__ == "__main__":
    main()
