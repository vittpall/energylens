#!/usr/bin/env python3
"""Figures for the AMD Instinct MI100 runs (needs test/amd_eval.py to have run).

  amd_characterization  energy per request, batch latency and power versus batch size, TP=1 and TP=2,
                        MI100 (solid) against A100 (dashed), one row per model (output limit 256)
  amd_tp_cost           TP=2 relative to TP=1 (energy, latency, power), MI100 against A100
  amd_accuracy          held-out MAPE per method at 50 rows, and the A100 -> MI100 transfer against
                        the number of MI100 rows
  amd_scatter           predicted against measured energy of a 50-row EnergyLens fit

    python viz/amd.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NVIDIA = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(NVIDIA, "test"))
sys.path.insert(0, os.path.join(NVIDIA, "..", "adaptive_sampling"))
from style import plt, style, save, LEGEND_SIZE  # noqa: E402
from amd_eval import RUNS, load, OUT  # noqa: E402
from compare_cross_model import fit_unified, unified_predict, holdout_scenarios  # noqa: E402

FIG = os.path.join(NVIDIA, "results_figures", "amd")
LAYOUT_COLOR = {1: "#4c72b0", 2: "#dd8452"}
METHOD_COLOR = {"EnergyLens": "#4c72b0", "MaverIQ": "#dd8452", "Gradient Boosting": "#55a868",
                "Linear Regression": "#8172b3", "Random Forest": "#c44e52", "Gaussian process": "#937860",
                "Log-linear": "#da8bc3"}
BINS = [0, 4, 8, 16, 32, 64, 128, 400]


def binned(d, tp, col, mt=256):
    s = d[(d.tp_degree == tp) & (d.pp_degree == 1) & (d.max_tokens == mt)]
    g = s.groupby(pd.cut(s.batch_size, BINS), observed=True).agg(b=("batch_size", "median"), y=(col, "median"))
    return g.b.to_numpy(), g.y.to_numpy()


def characterization():
    cols = [("measured_energy", "Energy per request (J)"), ("measured_latency", "Batch latency (s)"),
            ("gpu_power_w", "Total GPU power (W)")]
    fig, axes = plt.subplots(2, 3, figsize=(24, 12), sharex=True)
    for r, (name, (m, z)) in enumerate(RUNS.items()):
        mi, a1 = load(m), load(z)
        for c, (col, lab) in enumerate(cols):
            ax = axes[r, c]
            for tp in (1, 2):
                for d, ls, mk, hw in ((a1, "--", "s", "A100"), (mi, "-", "o", "MI100")):
                    x, y = binned(d, tp, col)
                    ax.plot(x, y, ls, marker=mk, color=LAYOUT_COLOR[tp], linewidth=3, markersize=9,
                            markeredgecolor="white", label=f"{hw}, TP={tp}")
            ax.set_xscale("log")
            if col != "gpu_power_w":
                ax.set_yscale("log")
            style(ax, "Batch size" if r == 1 else "", lab if True else "")
            if c == 0:
                ax.set_title(name, fontsize=26, loc="left")
    axes[0, 0].legend(fontsize=LEGEND_SIZE - 2, frameon=False, ncol=2, loc="upper right")
    save(fig, FIG, "amd_characterization")


def tp_cost():
    t = pd.read_csv(os.path.join(OUT, "tp_cost.csv"))
    g = t.groupby(["dataset", "hw"])[["energy_ratio", "latency_ratio", "power_ratio"]].median()
    labels = [("energy_ratio", "Energy"), ("latency_ratio", "Latency"), ("power_ratio", "Power")]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), sharey=True)
    for ax, name in zip(axes, RUNS):
        x = np.arange(3)
        for j, (hw, c) in enumerate((("A100", "#8da0cb"), ("MI100", "#e78ac3"))):
            v = [g.loc[(name, hw), k] for k, _ in labels]
            bars = ax.bar(x + (j - 0.5) * 0.38, v, 0.36, color=c, label=hw, edgecolor="white")
            for b_, val in zip(bars, v):
                ax.text(b_.get_x() + b_.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=18)
        ax.axhline(1, color="black", linewidth=1, linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in labels])
        ax.set_title(name, fontsize=24)
        ax.set_ylim(0, 1.95)
        style(ax, "", "TP=2 relative to TP=1" if ax is axes[0] else "")
    axes[0].legend(fontsize=LEGEND_SIZE, frameon=False, loc="upper left", ncol=2)
    save(fig, FIG, "amd_tp_cost")


def accuracy():
    a = pd.read_csv(os.path.join(OUT, "accuracy.csv"))
    tr = pd.read_csv(os.path.join(OUT, "transfer.csv"))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(22, 8), gridspec_kw={"width_ratios": [1.15, 1]})
    s = a[a.n == 50].groupby(["dataset", "method"]).mape.mean().unstack("method")
    order = [m for m in METHOD_COLOR if m in s.columns]
    x = np.arange(len(order))
    for j, name in enumerate(s.index):
        ax.bar(x + (j - 0.5) * 0.4, s.loc[name, order], 0.38, color=[METHOD_COLOR[m] for m in order],
               alpha=1.0 if j == 0 else 0.55, edgecolor="white", label=name)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in order], fontsize=15, rotation=35, ha="right")
    ax.set_ylim(5, 200)
    style(ax, "", "Held-out MAPE at 50 rows (%)")
    ax.text(0.98, 0.97, "solid: " + s.index[0] + "\nlight: " + s.index[1], transform=ax.transAxes, ha="right", va="top",
            fontsize=LEGEND_SIZE - 3)
    m = tr.groupby(["method", "k"]).mape.mean().unstack("k")
    ks = [5, 10, 25]
    for meth, c, mk in (("from scratch", "#4c72b0", "o"), ("A100 fit, rescaled", "#55a868", "s"),
                        ("A100 fit as prior", "#c44e52", "D")):
        bx.plot(ks, m.loc[meth, ks], "-", marker=mk, color=c, linewidth=3, markersize=10, label=meth.replace("A100", "A100"))
    bx.axhline(m.loc["A100 fit, as is", 0], color="#888888", linestyle="--", linewidth=2.5,
               label="A100 fit as is (0 rows)")
    bx.set_xscale("log")
    bx.set_xticks(ks)
    bx.set_xticklabels([str(k) for k in ks])
    bx.set_yscale("log")
    bx.set_yticks([10, 20, 40, 60])
    bx.set_yticklabels(["10", "20", "40", "60"])
    bx.minorticks_off()
    bx.set_ylim(7, 80)
    style(bx, "MI100 rows", "Held-out MAPE (%)")
    bx.legend(fontsize=LEGEND_SIZE - 3, frameon=False)
    save(fig, FIG, "amd_accuracy")


def scatter():
    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    for ax, (name, (m, _)) in zip(axes, RUNS.items()):
        d = load(m)
        pool, test, _, _ = holdout_scenarios(d, 0.3, 42)
        th = fit_unified(pool.sample(50, random_state=42), restarts=20)
        p = unified_predict(th, test)
        for tp in (1, 2):
            k = (test.tp_degree == tp).to_numpy()
            ax.scatter(test.measured_energy[k], p[k], s=36, alpha=0.6, color=LAYOUT_COLOR[tp], label=f"TP={tp}",
                       edgecolors="none")
        lim = [min(test.measured_energy.min(), p.min()) * 0.9, max(test.measured_energy.max(), p.max()) * 1.1]
        ax.plot(lim, lim, color="black", linewidth=1.5, linestyle=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(name, fontsize=24)
        style(ax, "Measured energy per request (J)", "Predicted (J)" if ax is axes[0] else "")
    axes[0].legend(fontsize=LEGEND_SIZE, frameon=False, loc="upper left", markerscale=2)
    save(fig, FIG, "amd_scatter")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    characterization()
    tp_cost()
    accuracy()
    scatter()
