#!/usr/bin/env python3
"""Evaluation figures of the paper (Sec. V.C) drawn from test-script outputs.

    python viz/evaluation.py no_power test_results/test_c_no_power.csv --out <dir>   # from nvidia/

  no_power   pairwise ranking without per-configuration power: EnergyLens vs MaverIQ with
             mean power (CSV from test/test_mean_power_accuracy.py)

Other evaluation figures are written by the test scripts themselves (learning curves:
test/compare_cross_model.py; trace MAPE: test/plot_trace.py; profiling cost:
adaptive_sampling/profiling_cost.py).
"""
import argparse

import numpy as np
import pandas as pd

from style import plt, style, save, LEGEND_SIZE


def no_power(csv, out):
    c = pd.read_csv(csv)
    c["label"] = c.dataset.str.replace(" (zeus)", "").str.replace("-fp16", "")
    p = c.pivot(index="label", columns="method", values="pairwise_acc").sort_values("EnergyLens")
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(p) + 1.6))   # one row per dataset
    y = np.arange(len(p))
    h = 0.38
    ax.barh(y + h / 2, p["EnergyLens"], h, color="#4c72b0", label="EnergyLens (no power input)")
    ax.barh(y - h / 2, p["MaverIQ_mean_P"], h, color="#dd8452", hatch="//", edgecolor="white",
            label="MaverIQ (mean power)")
    ax.axvline(50, color="#555555", linestyle=":", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(p.index)
    style(ax, "Pairwise ranking accuracy (%)")
    ax.tick_params(axis="y", labelsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(p) - 0.3)
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, out, "no_power")


FIGURES = {"no_power": no_power}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("figure", choices=list(FIGURES))
    ap.add_argument("csv")
    ap.add_argument("--out", default="results_figures/evaluation")
    args = ap.parse_args()
    FIGURES[args.figure](args.csv, args.out)


if __name__ == "__main__":
    main()
