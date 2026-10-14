"""Compare sampling strategies on the Zeus grids at a fixed row budget (and optional time budget).

    python run.py                                  # all datasets, strategies, 3 seeds, 50 rows
    python run.py --strategies random doptimal_cost --seeds 42
    python run.py --time-budget 600                # also stop each strategy after 10 profiling minutes
    python run.py --all --out results_all          # every dataset of the paper (nvidia/viz/datasets.py)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import itertools
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulator import DATASETS, PATHS, Simulator, evaluate
from strategies import STRATEGIES


def run_one(task):
    name, strategy, seed, n, time_budget = task
    sim = Simulator(name, seed)
    if strategy == "full_pool":
        sim.measure(sim.available())
    else:
        rng = np.random.RandomState(seed)
        for rows in STRATEGIES[strategy](sim, n, rng):
            done = sim.measure(rows[: n - len(sim.measured)], limit=time_budget)
            if not done or len(sim.measured) >= n:
                break
    return {"dataset": name, "strategy": strategy, "seed": seed, "rows": len(sim.measured),
            "engines": len({sim.engine[i] for i in sim.measured}), "reloads": sim.reloads,
            "minutes": sim.cost / 60, **evaluate(sim.observed(), sim.test)}


def plot(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, metric, label in [(axes[0], "pairwise", "Pairwise accuracy (%)"),
                              (axes[1], "mape", "MAPE (%)")]:
        for i, (name, r) in enumerate(summary.iterrows()):
            ax.errorbar(r[("minutes", "mean")], r[(metric, "mean")], yerr=r[(metric, "std")],
                        fmt="o", ms=9, color=f"C{i}", mec="white", mew=0.8, capsize=3, label=name)
        ax.set_xlabel("Profiling time (min)", fontsize=15, fontweight="bold")
        ax.set_ylabel(label, fontsize=15, fontweight="bold")
        ax.tick_params(labelsize=13)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_facecolor("#f9f9f9")
    axes[0].legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(PATHS))
    ap.add_argument("--all", action="store_true", help="every dataset of the paper's registry")
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--n", type=int, default=50, help="row budget")
    ap.add_argument("--time-budget", type=float, default=None, help="profiling-seconds budget")
    ap.add_argument("--no-full-pool", action="store_true", help="skip the all-rows reference")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count()))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()

    if args.all:
        args.datasets = list(PATHS)
    limit = args.time_budget if args.time_budget else np.inf
    strategies = args.strategies + ([] if args.no_full_pool else ["full_pool"])
    tasks = [(d, s, seed, args.n, limit)
             for d, s, seed in itertools.product(args.datasets, strategies, args.seeds)]
    with mp.Pool(args.workers) as pool:
        runs = pd.DataFrame(pool.map(run_one, tasks, chunksize=1))

    os.makedirs(args.out, exist_ok=True)
    tag = f"n{args.n}" + (f"_t{int(args.time_budget)}s" if args.time_budget else "")
    runs.to_csv(os.path.join(args.out, f"runs_{tag}.csv"), index=False)

    cols = ["rows", "engines", "reloads", "minutes", "mape", "pairwise", "top1", "regret"]
    per_seed = runs.groupby(["strategy", "seed"])[cols].mean()
    summary = per_seed.groupby("strategy").agg(["mean", "std"]).reindex(strategies)
    summary.to_csv(os.path.join(args.out, f"summary_{tag}.csv"))
    by_dataset = runs.groupby(["dataset", "strategy"])[cols].mean().round(2)
    by_dataset.to_csv(os.path.join(args.out, f"by_dataset_{tag}.csv"))

    fmt = lambda c: summary[(c, "mean")].map("{:.1f}".format) + " ± " + summary[(c, "std")].map("{:.1f}".format)
    print(pd.DataFrame({c: fmt(c) for c in ["rows", "reloads", "minutes", "mape", "pairwise", "top1", "regret"]}).to_string())
    plot(summary.drop(index="full_pool", errors="ignore"), os.path.join(args.out, f"tradeoff_{tag}.png"))


if __name__ == "__main__":
    main()
