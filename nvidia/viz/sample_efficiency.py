#!/usr/bin/env python3
"""Box plots of held-out MAPE and Top-1 accuracy versus the number of training rows.

One box per (method, n) over the 29 A100 datasets x five seeds (test/sample_efficiency.py):
box = 25th-75th percentile, line = median, whiskers = 5th-95th percentile.

    python viz/sample_efficiency.py
"""
import os

import numpy as np
import pandas as pd

from style import plt, style, save, LEGEND_SIZE

HERE = os.path.dirname(os.path.abspath(__file__))
NVIDIA = os.path.dirname(HERE)
METHODS = [("EnergyLens", "EnergyLens", "#4c72b0"), ("MaverIQ", "MaverIQ", "#dd8452"),
           ("Gradient Boosting", "Gradient Boosting", "#55a868"), ("Linear Regression", "Linear Regression", "#8172b3"),
           ("Random Forest", "Random Forest", "#c44e52")]


def panel(ax, r, metric, ylabel, log):
    ns = sorted(r.n.unique())
    w = 0.8 / len(METHODS)
    for j, (m, label, c) in enumerate(METHODS):
        data = [r[(r.method == m) & (r.n == n)][metric].dropna().to_numpy() for n in ns]
        pos = np.arange(len(ns)) + (j - (len(METHODS) - 1) / 2) * w
        bp = ax.boxplot(data, positions=pos, widths=w * 0.9, whis=(5, 95), showfliers=False, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5))
        for b in bp["boxes"]:
            b.set(facecolor=c, alpha=0.85, edgecolor="#333333")
        ax.plot([], [], "s", color=c, markersize=12, label=label)
    ax.set_xticks(np.arange(len(ns)))
    ax.set_xticklabels([str(n) for n in ns])
    if log:
        ax.set_yscale("log")
    style(ax, "Training rows", ylabel)


def main():
    r = pd.read_csv(os.path.join(NVIDIA, "results_figures", "sample_efficiency", "sample_efficiency.csv"))
    fig, (a, b) = plt.subplots(2, 1, figsize=(17, 11.0), sharex=True)
    panel(a, r, "mape", "Held-out MAPE (%)", log=True)
    a.set_xlabel("")
    panel(b, r, "top1", "Top-1 accuracy (%)", log=False)
    b.set_ylim(0, 102)
    a.legend(fontsize=LEGEND_SIZE - 2, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=5, frameon=False)
    save(fig, os.path.join(NVIDIA, "results_figures", "evaluation"), "sample_efficiency_box")


if __name__ == "__main__":
    main()


def bands():
    """Option 2: spread across workloads (band) separated from stability across seeds (third panel)."""
    r = pd.read_csv(os.path.join(NVIDIA, "results_figures", "sample_efficiency", "sample_efficiency.csv"))
    per = r.groupby(["method", "n", "dataset"]).agg(mape=("mape", "mean"), top1=("top1", "mean"),
                                                    sd=("mape", "std")).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.0))
    for m, label, c in METHODS:
        g = per[per.method == m].groupby("n")
        ns = np.array(sorted(per.n.unique()))
        for ax, col in zip(axes[:2], ["mape", "top1"]):
            q = g[col].quantile([0.25, 0.5, 0.75]).unstack()
            ax.plot(ns, q[0.5], "o-", color=c, linewidth=2.5, markersize=8, label=label)
            ax.fill_between(ns, q[0.25], q[0.75], color=c, alpha=0.2)
        axes[2].plot(ns, g["sd"].median(), "o-", color=c, linewidth=2.5, markersize=8, label=label)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
    axes[0].set_yscale("log")
    axes[2].set_yscale("log")
    style(axes[0], "Training rows", "Held-out MAPE (%)")
    style(axes[1], "Training rows", "Top-1 accuracy (%)")
    style(axes[2], "Training rows", "MAPE std. over seeds (pts)")
    axes[1].set_ylim(0, 102)
    axes[0].legend(fontsize=LEGEND_SIZE, loc="upper right", frameon=False)
    save(fig, os.path.join(NVIDIA, "results_figures", "evaluation"), "sample_efficiency_bands")


if __name__ == "__main__" and os.environ.get("BANDS"):
    bands()
