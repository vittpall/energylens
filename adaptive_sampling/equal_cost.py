#!/usr/bin/env python3
"""Equal-cost comparison of profiling designs, scored where the layout choice is not trivial.

    python equal_cost.py [--seeds 42 7 123] [--sizes 12 25 50]
"""
import argparse
import multiprocessing as mp
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from simulator import PATHS, RELOAD_S, Simulator, ranking_metrics  # noqa: E402
from strategies import STRATEGIES  # noqa: E402
from profiling_cost import (charge, reference_rows, fitted_exponents, maveriq_energy_closed_form,  # noqa: E402
                            fit_unified, unified_predict)

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = ["max_tokens", "batch_size"]


def group(name):
    n = name.lower()
    if "4-bit" in n or "bnb" in n:
        return "Dense, 4-bit"
    if "mamba" in n or "nemotron" in n:
        return "State-space, hybrid"
    if "moe" in n or "mixtral" in n:
        return "Mixture of experts"
    if "video-only" in n:
        return "Video only"
    if "video" in n:
        return "Video+text"
    if "image" in n:
        return "Image+text"
    return "Dense, text"


def per_config(test, pred):
    t = test.assign(pred=pred, n=test.tp_degree * test.pp_degree)
    return t.groupby(SCEN + ["tp_degree", "pp_degree"]).agg(
        y=("measured_energy", "mean"), p=("pred", "mean"), lat=("measured_latency", "mean"),
        n=("n", "first")).reset_index()


def fixed_gpu(c):
    """Regret / pairwise among layouts with the same GPU count, per scenario."""
    reg, pair = [], []
    for _, g in c.groupby(SCEN + ["n"]):
        if len(g) < 2:
            continue
        y, p = g.y.to_numpy(), g.p.to_numpy()
        reg.append(100 * (y[p.argmin()] / y.min() - 1))
        dy, dp = y[:, None] - y[None, :], p[:, None] - p[None, :]
        v = np.triu(np.abs(dy) > 1e-9, 1)
        if v.any():
            pair.append(((dy * dp) > 0)[v].mean())
    return {"fixed_regret": np.mean(reg) if reg else np.nan, "fixed_pairwise": 100 * np.mean(pair) if pair else np.nan,
            "fixed_scen": len(reg)}


def slo(c, k):
    reg, rule_breaks, rule_reg = [], [], []
    for _, g in c.groupby(SCEN):
        if len(g) < 2:
            continue
        f = g[g.lat <= k * g.lat.min() + 1e-12]
        best = f.y.min()
        reg.append(100 * (f.y.to_numpy()[f.p.to_numpy().argmin()] / best - 1))
        rule = g.sort_values(["n", "pp_degree"]).iloc[0]
        rule_breaks.append(rule.lat > k * g.lat.min() + 1e-12)
        rf = f.sort_values(["n", "pp_degree"]).iloc[0]            # fewest GPUs among the compliant
        rule_reg.append(100 * (rf.y / best - 1))
    return {f"slo{k}_regret": np.mean(reg), f"slo{k}_rule_breaks": 100 * np.mean(rule_breaks),
            f"slo{k}_rule_regret": np.mean(rule_reg)}


def score(test, pred):
    y = test.measured_energy.to_numpy(float)
    c = per_config(test, pred)
    out = {"mape": 100 * np.mean(np.abs(pred - y) / y), **ranking_metrics(test, pred), **fixed_gpu(c)}
    for k in (1.5, 2.0):
        out.update(slo(c, k))
    return out


def collect(name, seed, strategy, n):
    sim = Simulator(name, seed)
    rng = np.random.RandomState(seed)
    for rows in STRATEGIES[strategy](sim, n, rng):
        sim.measure(rows[: n - len(sim.measured)])
        if len(sim.measured) >= n:
            break
    return sim, sim.observed()


def run_one(task):
    name, seed, sizes, with_maveriq = task
    out = []
    try:
        reload_s = RELOAD_S[name]
        add = lambda m, n, tr, test, pred: out.append({"dataset": name, "group": group(name), "seed": seed,
                                                       "method": m, "n": n, **charge(tr, reload_s), **score(test, pred)})
        if with_maveriq:
            try:
                sim = Simulator(name, seed)
                ref = sim.pool.iloc[reference_rows(sim)]
                exps = fitted_exponents(name, seed)
                add("MaverIQ (its 12 rows)", 12, ref, sim.test, maveriq_energy_closed_form(ref, sim.test, exps))
                add("EnergyLens, MaverIQ's 12 rows", 12, ref, sim.test, unified_predict(fit_unified(ref), sim.test))
            except Exception as e:      # its reference rows need batch sizes shared by both output lengths
                print(f"[no MaverIQ] {name} seed {seed}: {type(e).__name__}: {e}", flush=True)
        for n in sizes:
            for label, strat in (("random", "random"), ("random configs", "random_engines"),
                                 ("D-optimal", "doptimal_cost")):
                s, tr = collect(name, seed, strat, n)
                add(f"EnergyLens, {label}", n, tr, s.test, unified_predict(fit_unified(tr), s.test))
    except Exception as e:                                          # keep the other datasets
        print(f"[skip] {name} seed {seed}: {type(e).__name__}: {e}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", nargs="+", default=list(PATHS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--sizes", type=int, nargs="+", default=[12, 25, 50])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "equal_cost"))
    ap.add_argument("--no-maveriq", action="store_true", help="skip MaverIQ's 12-row protocol")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("NCPUS", os.cpu_count())),
                    help="worker processes (default: the job's allocated CPUs)")
    a = ap.parse_args()
    # Results are appended per (dataset, seed) as they finish, so a job that hits its
    # walltime keeps what it computed and a resubmit skips it.
    path = f"{a.out}.csv"
    done = set()
    if os.path.exists(path):
        prev = pd.read_csv(path)
        done = set(zip(prev.dataset, prev.seed))
    tasks = [(d, s, a.sizes, not a.no_maveriq) for d in a.datasets for s in a.seeds if (d, s) not in done]
    print(f"{len(done)} (dataset, seed) pairs already in {path}; {len(tasks)} to run", flush=True)
    # One worker per allocated CPU: a worker per node core exhausted the job's memory, a
    # killed worker left multiprocessing.Pool waiting forever. ProcessPoolExecutor raises.
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=max(1, min(a.workers, len(tasks) or 1))) as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        for f in as_completed(futs):
            rows = f.result()
            if rows:
                pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
                print(f"saved {rows[0]['dataset']} seed {rows[0]['seed']}", flush=True)
    r = pd.read_csv(path)
    cols = ["gpu_hours", "minutes", "engines", "mape", "pairwise", "top1", "regret", "fixed_pairwise",
            "fixed_regret", "slo1.5_regret", "slo2.0_regret", "slo2.0_rule_regret", "slo2.0_rule_breaks"]
    pd.set_option("display.width", 250)
    per_ds = r.groupby(["method", "n", "dataset"])[cols].mean()
    print(f"{r.dataset.nunique()} datasets x {r.seed.nunique()} seeds; mean over datasets")
    print(per_ds.groupby(["method", "n"]).mean().round(2).to_string())
    print("\nby group (n = 12 MaverIQ vs EnergyLens D-optimal 25 / 50): regret, fixed-GPU regret, SLO 2x regret")
    g = r[(r.method.str.startswith("MaverIQ")) | ((r.method == "EnergyLens, D-optimal") & r.n.isin([25, 50]))]
    print(g.groupby(["group", "method", "n"])[["gpu_hours", "regret", "fixed_regret", "slo2.0_regret"]]
          .mean().round(2).to_string())


if __name__ == "__main__":
    main()
