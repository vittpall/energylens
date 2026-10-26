#!/usr/bin/env python3
"""EnergyLens on AMD Instinct MI100 (2 GPUs, PCIe, vLLM 0.6.3), against the A100 results.

The node has two GPUs and vLLM 0.6.3's synchronous engine has no pipeline parallelism, so the layouts
are (TP, PP) = (1,1) and (2,1) only: a scenario has a single pair of layouts to order. Pairwise accuracy
and Top-1 are therefore easy here and are not comparable with the A100's six layouts; the extra
metric "ratio error" scores the quantity the choice depends on, the energy of TP=2 relative to TP=1.

  0. validity   rows, layouts, power source, energy ~ power x latency / batch, output length per layout
  1. accuracy   n-row fits (n = 25, 50, 100), scored on held-out scenarios (5 seeds): EnergyLens, MaverIQ,
                Gradient Boosting, Linear Regression, Random Forest, Gaussian process, log-linear
  2. optimum    latency- vs energy-optimal layout per scenario
  3. transfer   the A100 fit of the same model applied on MI100: as is, rescaled on k MI100 rows, or
                refitted from it with a prior (k = 5, 10, 25), against a fit from scratch on the same rows
  4. exponents  the fitted exponents on both machines (with two TP levels, gamma_TP and delta_1 are only
                weakly separated; read the exponents as indicative)
  5. TP cost    energy, latency and power of TP=2 relative to TP=1, MI100 against A100, by batch-size bin

    python test/amd_eval.py
Writes results_figures/amd/*.csv.
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
from shift_transfer import fit_all as _fit_all, mape, fit_scale, fit_prior, gb_fit  # noqa: E402
from phi_and_baselines import loglinear_fit  # noqa: E402
from more_baselines import gp_fit  # noqa: E402
from sample_efficiency import sk_fit  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

OUT = os.path.join(NVIDIA, "results_figures", "amd")
M = "results_amd/run_text_only_{}_fp16/res_native/profiling_metrics.csv"
Z = "results_zeus/run_text_only_{}_fp16/res_native/profiling_metrics.csv"
RUNS = {"Mistral-7B": (M.format("Mistral-7B-Instruct-v0.3"), Z.format("Mistral-7B-Instruct-v0.3")),
        "Llama-3.1-8B": (M.format("Llama-3.1-8B-Instruct"), Z.format("Llama-3.1-8B-Instruct"))}
SEEDS = (42, 7, 123, 1, 2)
NS = (25, 50, 100)
NAMES = ["alpha_p", "beta_p", "gamma1_p", "gamma2_p", "eps_p", "alpha_d", "beta_d", "gamma1_d", "gamma2_d",
         "eps_d", "delta1", "delta2"]
SC = ["max_tokens", "batch_size"]


def load(rel):
    return load_csv(os.path.join(NVIDIA, rel))


def fit_every(train):
    fns = _fit_all(train)
    fns["Linear Regression"] = sk_fit(train, LinearRegression())
    fns["Random Forest"] = sk_fit(train, RandomForestRegressor(n_estimators=100, random_state=42))
    fns["Gaussian process"] = gp_fit(train)
    fns["Log-linear"] = loglinear_fit(train)
    return fns


def ratio_error(test, p):
    """Median error (%) of the predicted energy of (2,1) relative to (1,1), per held-out scenario."""
    t = test.assign(pred=np.asarray(p, float))
    m = t.groupby(SC + ["tp_degree"])[["measured_energy", "pred"]].mean().unstack("tp_degree").dropna()
    if m.empty or ("measured_energy", 2) not in m or ("measured_energy", 1) not in m:
        return np.nan
    true = m[("measured_energy", 2)] / m[("measured_energy", 1)]
    pred = m[("pred", 2)] / m[("pred", 1)]
    return float(100 * np.median(np.abs(pred / true - 1)))


def validity(name, rel):
    r = pd.read_csv(os.path.join(NVIDIA, rel))
    n_raw = len(r)
    ok = r.dropna(subset=["energy", "power", "latency"])
    ok = ok[pd.to_numeric(ok["energy"], errors="coerce") > 0]
    rat = ok.energy * ok.batch_size / (ok.latency * ok.power)
    lay = list(zip(ok.tensor_parallel_size, ok.pipeline_parallel_size))
    ln = (ok.max_generated_tokens / ok.max_tokens).groupby(pd.Series(lay, index=ok.index)).mean().round(2).to_dict()
    return {"dataset": name, "raw rows": n_raw, "valid rows": len(load(rel)),
            "power source": ",".join(sorted(r.power_source.dropna().unique())),
            "layouts": str(sorted(set(lay))), "median E*b/(L*P)": round(float(rat.median()), 3),
            "rows outside +-10%": round(100 * float(((rat < 0.9) | (rat > 1.1)).mean()), 1),
            "generated / max tokens": str(ln)}


def accuracy(data):
    rows = []
    for name, d in data.items():
        for seed in SEEDS:
            pool, test, _, _ = holdout_scenarios(d, 0.3, seed)
            y = test["measured_energy"].to_numpy(float)
            for n in NS:
                if n > len(pool):
                    continue
                for meth, f in fit_every(pool.sample(n, random_state=seed)).items():
                    p = np.asarray(f(test), float)
                    rows.append({"dataset": name, "seed": seed, "n": n, "method": meth, "mape": mape(p, y),
                                 "ratio_err": ratio_error(test, p), **ranking_metrics(test, p)})
        print(f"[accuracy] {name} done", flush=True)
    return pd.DataFrame(rows)


def optimum(data):
    rows = []
    for name, d in data.items():
        g = d.groupby(SC + ["tp_degree", "pp_degree"])[["measured_latency", "measured_energy"]].mean().reset_index()
        for _, s in g.groupby(SC):
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
        th_m = fit_unified(d, restarts=10)
        thetas += [{"dataset": name, "gpu": "A100", **dict(zip(NAMES, th_a))},
                   {"dataset": name, "gpu": "MI100", **dict(zip(NAMES, th_m))}]
        for seed in SEEDS:
            pool, test, _, _ = holdout_scenarios(d, 0.3, seed)
            y = test["measured_energy"].to_numpy(float)
            p0 = unified_predict(th_a, test)
            rows.append({"dataset": name, "seed": seed, "k": 0, "method": "A100 fit, as is", "mape": mape(p0, y),
                         "ratio_err": ratio_error(test, p0), **ranking_metrics(test, p0)})
            for k in ks:
                tr = pool.sample(k, random_state=seed)
                preds = {"A100 fit, rescaled": fit_scale(th_a, tr) * p0,
                         "A100 fit as prior": unified_predict(fit_prior(tr, th_a, 0.1), test),
                         "from scratch": unified_predict(fit_unified(tr, restarts=10), test),
                         "Gradient Boosting": gb_fit(tr)(test)}
                for m, p in preds.items():
                    p = np.asarray(p, float)
                    rows.append({"dataset": name, "seed": seed, "k": k, "method": m, "mape": mape(p, y),
                                 "ratio_err": ratio_error(test, p), **ranking_metrics(test, p)})
        print(f"[transfer] {name} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(thetas)


def tp_cost(data, a100):
    """TP=2 relative to TP=1 (energy per request, batch latency, power), by batch-size bin and output limit."""
    bins = [0, 8, 16, 32, 64, 400]
    rows = []
    for name in data:
        for hw, d in (("A100", a100[name]), ("MI100", data[name])):
            g = d.assign(bin=pd.cut(d.batch_size, bins)).groupby(["max_tokens", "bin", "tp_degree"], observed=True)[
                ["measured_energy", "measured_latency", "gpu_power_w"]].mean().unstack("tp_degree").dropna()
            for (mt, b), r in g.iterrows():
                if (hw == "A100") and (("measured_energy", 2) not in r.index):
                    continue
                rows.append({"dataset": name, "hw": hw, "max_tokens": mt, "bin": str(b),
                             "energy_ratio": r[("measured_energy", 2)] / r[("measured_energy", 1)],
                             "latency_ratio": r[("measured_latency", 2)] / r[("measured_latency", 1)],
                             "power_ratio": r[("gpu_power_w", 2)] / r[("gpu_power_w", 1)]})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 220)
    data = {n: load(m) for n, (m, _) in RUNS.items()}
    a100 = {n: load(z) for n, (_, z) in RUNS.items()}

    v = pd.DataFrame([validity(n, m) for n, (m, _) in RUNS.items()])
    v.to_csv(os.path.join(OUT, "validity.csv"), index=False)
    print("== 0. validity"); print(v.T.to_string())

    acc = accuracy(data)
    acc.to_csv(os.path.join(OUT, "accuracy.csv"), index=False)
    print("\n== 1. accuracy on held-out scenarios (mean over seeds)")
    for n in NS:
        print(f"-- n = {n}")
        print(acc[acc.n == n].groupby(["method"])[["mape", "pairwise", "top1", "regret", "ratio_err"]].mean().round(2).to_string())

    op = optimum(data)
    op.to_csv(os.path.join(OUT, "optimum.csv"), index=False)
    print("\n== 2. latency- vs energy-optimal layout")
    print(op.groupby("dataset").agg(scenarios=("differ", "size"), differ=("differ", "mean"), penalty=("penalty", "mean")).round(3).to_string())
    print(op.groupby("dataset")[["fast", "cheap"]].agg(lambda s: s.value_counts().index[0]).to_string())

    tr, th = transfer(data, a100)
    tr.to_csv(os.path.join(OUT, "transfer.csv"), index=False)
    th.to_csv(os.path.join(OUT, "theta.csv"), index=False)
    print("\n== 3. A100 -> MI100 transfer (mean over models and seeds)")
    for m in ("mape", "ratio_err", "pairwise", "regret"):
        print(f"-- {m}"); print(tr.pivot_table(index="method", columns="k", values=m).round(2).to_string())
    print("\n== 4. fitted parameters")
    print(th.set_index(["dataset", "gpu"])[["beta_p", "gamma1_p", "beta_d", "gamma1_d", "gamma2_d", "alpha_d", "delta1"]].round(2).to_string())

    tc = tp_cost(data, a100)
    tc.to_csv(os.path.join(OUT, "tp_cost.csv"), index=False)
    print("\n== 5. TP=2 relative to TP=1 (median over bins)")
    print(tc.groupby(["dataset", "hw"])[["energy_ratio", "latency_ratio", "power_ratio"]].median().round(2).to_string())


if __name__ == "__main__":
    main()
