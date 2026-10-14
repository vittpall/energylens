#!/usr/bin/env python3
"""
Seed Variance Analysis
======================
Shows how much MAPE / R² varies across random training samples for each
model configuration (dataset × n_train). Every method is scored on a held-out
test set (--test-frac of the rows, drawn per seed), never on its training rows.  Unlike compare_cross_model.py,
this script keeps every seed result instead of averaging them, so you can
see the spread (std, min, max) introduced by the random sample draw.

Usage:
    python seed_variance.py                         # all datasets, default seeds
    python seed_variance.py --n-seeds 10            # more seeds for tighter estimates
    python seed_variance.py --n-train 50 100 500
    python seed_variance.py --datasets "Mistral-7B (NVIDIA A100)"
    python seed_variance.py --output-dir variance_out
"""

import argparse, os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ── reuse helpers from the sibling module ──────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from compare_cross_model import (
    DATASETS, prepare_df,
    fit_unified, unified_predict,
    fit_maveriq, maveriq_predict,
    fit_predict_ml,
    compute_metrics,
    holdout_rows, TEST_FRAC,
    _SKLEARN,
)
from dataset_args import add_datasets_arg, resolve_datasets

N_SEEDS_DEFAULT = 10
N_TRAIN_DEFAULT = [50, 100, 500]


# ════════════════════════════════════════════════════════════
#  COLLECT PER-SEED RESULTS
# ════════════════════════════════════════════════════════════

def collect_seed_results(df, n_train, n_seeds, test_frac=TEST_FRAC):
    """
    Return dict[method] = list of (mape, r2, rmse) — one entry per seed.

    Per seed, a held-out test set (test_frac of the rows) is drawn first; the
    n_train training rows come from the remaining pool and every method is
    scored on the test rows only. So the spread across seeds covers both the
    training draw and the test draw. test_frac=0 restores the old behaviour
    (score on the full df, training rows included). Returns None when n_train
    exceeds the training pool.
    """
    runs = {}

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed * 7 + 13)
        pool_df, test_df = holdout_rows(df, rng, test_frac)
        if test_frac and test_frac > 0 and n_train > len(pool_df):
            return None
        if n_train >= len(pool_df):
            train_df = pool_df
        else:
            idx      = rng.choice(len(pool_df), size=n_train, replace=False)
            train_df = pool_df.iloc[idx].reset_index(drop=True)
        true_e = test_df['measured_energy'].values.astype(float)

        # EnergyLens
        try:
            p    = fit_unified(train_df)
            pred = unified_predict(p, test_df)
            m    = compute_metrics(pred, true_e)
        except Exception as e:
            print(f"    [WARN] EnergyLens seed {seed}: {e}")
            m = (np.nan, np.nan, np.nan)
        runs.setdefault('EnergyLens (Ours)', []).append(m)

        # MaverIQ
        try:
            p    = fit_maveriq(train_df)
            pred = maveriq_predict(p, test_df)
            m    = compute_metrics(pred, true_e)
        except Exception as e:
            print(f"    [WARN] MaverIQ seed {seed}: {e}")
            m = (np.nan, np.nan, np.nan)
        runs.setdefault('MaverIQ (Lat×P)', []).append(m)

        # ML baselines
        ml = fit_predict_ml(train_df, test_df, true_e)
        for name, m in ml.items():
            runs.setdefault(name, []).append(m)

    return runs   # dict[method] -> list[(mape, r2, rmse)]


def summarise(runs):
    """Compute mean ± std, min, max per method from seed runs."""
    summary = {}
    for method, metrics_list in runs.items():
        arr = np.array(metrics_list, dtype=float)   # shape (n_seeds, 3)
        summary[method] = {
            'mean': np.nanmean(arr, axis=0),
            'std':  np.nanstd(arr,  axis=0),
            'min':  np.nanmin(arr,  axis=0),
            'max':  np.nanmax(arr,  axis=0),
            'raw':  arr,
        }
    return summary


# ════════════════════════════════════════════════════════════
#  PRINT TABLE
# ════════════════════════════════════════════════════════════

def print_variance_table(all_summaries):
    """
    all_summaries[ds][n_train][method] = summary dict
    """
    print(f"\n{'='*100}")
    print("  VARIANCE TABLE — MAPE mean ± std  (min … max)  across random training samples")
    print(f"{'='*100}")

    for ds, by_n in all_summaries.items():
        print(f"\n  ── {ds} ──")
        for n_train, by_m in sorted(by_n.items()):
            print(f"\n    n_train = {n_train}")
            print(f"    {'Method':<28}  {'MAPE mean':>9}  {'±std':>7}  {'min':>7}  {'max':>7}  "
                  f"{'R² mean':>8}  {'±std':>6}")
            print(f"    {'-'*85}")
            for method, s in by_m.items():
                mape_m, r2_m = s['mean'][0], s['mean'][1]
                mape_s, r2_s = s['std'][0],  s['std'][1]
                mape_min     = s['min'][0]
                mape_max     = s['max'][0]
                print(f"    {method:<28}  {mape_m:>8.2f}%  {mape_s:>6.2f}%  "
                      f"{mape_min:>6.2f}%  {mape_max:>6.2f}%  "
                      f"{r2_m:>8.4f}  {r2_s:>6.4f}")


# ════════════════════════════════════════════════════════════
#  PLOTS
# ════════════════════════════════════════════════════════════

COLORS = {
    'EnergyLens (Ours)': '#2a9d8f',
    'Linear Regression':  '#457b9d',
    'Random Forest':      '#e9c46a',
    'Gradient Boosting':  '#f4a261',
    'MaverIQ (Lat×P)':   '#e76f51',
}


def plot_boxplots(all_summaries, output_dir):
    """
    One figure per dataset: side-by-side box plots for each n_train,
    showing MAPE distribution across seeds for every method.
    """
    plt.ioff()
    for ds, by_n in all_summaries.items():
        n_trains = sorted(by_n.keys())
        methods  = list(next(iter(by_n.values())).keys())
        n_m      = len(methods)
        fig, axes = plt.subplots(1, len(n_trains),
                                 figsize=(5 * len(n_trains), 5),
                                 sharey=False)
        if len(n_trains) == 1:
            axes = [axes]

        for ax, n_train in zip(axes, n_trains):
            data   = [by_n[n_train][m]['raw'][:, 0] for m in methods]  # MAPE per seed
            bplot  = ax.boxplot(data, patch_artist=True, notch=False,
                                medianprops=dict(color='black', lw=2))
            for patch, method in zip(bplot['boxes'], methods):
                patch.set_facecolor(COLORS.get(method, 'steelblue'))
                patch.set_alpha(0.7)
            ax.set_xticks(range(1, n_m + 1))
            ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
            ax.set_title(f'n_train = {n_train}', fontsize=10, fontweight='bold')
            ax.set_ylabel('MAPE (%)')
            ax.grid(axis='y', ls='--', alpha=0.35)

        slug = ds.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        fig.suptitle(f'Seed Variance — {ds}', fontsize=12, fontweight='bold')
        fig.tight_layout()
        out = os.path.join(output_dir, f'variance_boxplot_{slug}.png')
        fig.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {out}")


def plot_errorbars(all_summaries, output_dir):
    """
    One figure per dataset: MAPE mean ± std vs n_train for each method.
    """
    plt.ioff()
    for ds, by_n in all_summaries.items():
        n_trains = sorted(by_n.keys())
        methods  = list(next(iter(by_n.values())).keys())

        fig, ax = plt.subplots(figsize=(7, 4))
        for method in methods:
            means = [by_n[n][method]['mean'][0] for n in n_trains]
            stds  = [by_n[n][method]['std'][0]  for n in n_trains]
            ax.errorbar(n_trains, means, yerr=stds,
                        label=method,
                        color=COLORS.get(method, 'grey'),
                        marker='o', lw=1.8, capsize=4, capthick=1.5,
                        ls='-' if method in ('EnergyLens (Ours)', 'MaverIQ (Lat×P)') else '--')

        ax.set_xlabel('n_train (training samples)', fontsize=11)
        ax.set_ylabel('MAPE (%) — mean ± std across seeds', fontsize=11)
        ax.set_title(f'Sample Variance — {ds}', fontsize=11, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xticks(n_trains)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(ls='--', alpha=0.35)
        fig.tight_layout()

        slug = ds.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        out  = os.path.join(output_dir, f'variance_errorbars_{slug}.png')
        fig.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {out}")


def plot_std_heatmap(all_summaries, output_dir):
    """
    Heatmap of MAPE std: rows = datasets, cols = (method, n_train).
    High std = unreliable under different random samples.
    """
    plt.ioff()
    datasets = list(all_summaries.keys())
    n_trains = sorted(next(iter(all_summaries.values())).keys())
    methods  = list(next(iter(next(iter(all_summaries.values())).values())).keys())

    cols = [(m, n) for n in n_trains for m in methods]
    data = np.full((len(datasets), len(cols)), np.nan)

    for ri, ds in enumerate(datasets):
        for ci, (m, n) in enumerate(cols):
            if n in all_summaries[ds] and m in all_summaries[ds][n]:
                data[ri, ci] = all_summaries[ds][n][m]['std'][0]  # MAPE std

    col_labels = [f"{m}\nn={n}" for m, n in cols]
    fig, ax = plt.subplots(figsize=(max(16, len(cols) * 1.4), max(5, len(datasets) * 0.8)))
    vmax = np.nanpercentile(data, 95) if not np.all(np.isnan(data)) else 1.0
    im = ax.imshow(data, cmap='YlOrRd', vmin=0, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax, label='MAPE std (%)')
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha='right')
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=9)
    for ri in range(len(datasets)):
        for ci in range(len(cols)):
            v = data[ri, ci]
            if not np.isnan(v):
                tc = 'white' if v / max(vmax, 1e-9) > 0.6 else 'black'
                ax.text(ci, ri, f'{v:.1f}', ha='center', va='center', fontsize=6, color=tc)
    ax.set_title('MAPE Std Heatmap — Variance from Random Training Samples', fontsize=12, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(output_dir, 'variance_std_heatmap.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out}")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='Seed variance analysis for model fitting')
    ap.add_argument('--output-dir', default='results/seed_variance',
                    help='Directory to save CSV and plots')
    ap.add_argument('--n-train',    type=int, nargs='+', default=N_TRAIN_DEFAULT)
    ap.add_argument('--n-seeds',    type=int, default=N_SEEDS_DEFAULT,
                    help='Number of random seeds (more = tighter estimates)')
    ap.add_argument('--no-plots',   action='store_true')
    add_datasets_arg(ap)
    ap.add_argument('--test-frac',  type=float, default=TEST_FRAC,
                    help='Fraction of rows held out per seed for scoring '
                         '(default 0.2); 0 = score on the full dataset')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    datasets = resolve_datasets(args.datasets, DATASETS)
    missing = [(n, p) for n, p in datasets.items() if not os.path.isfile(p)]
    if missing:
        print('[WARNING] Missing datasets:')
        for n, p in missing:
            print(f'  {n}: {p}')
        datasets = {k: v for k, v in datasets.items() if (k, v) not in missing}
    if not datasets:
        print('[ERROR] No datasets available.'); return

    all_summaries = {}   # [ds][n_train][method] = summary dict
    csv_rows      = []

    for ds_name, ds_path in datasets.items():
        print(f"\n{'━'*70}")
        print(f"  Dataset : {ds_name}")
        raw = pd.read_csv(ds_path)
        print(f"  Rows    : {len(raw)}")
        try:
            df = prepare_df(raw, ds_path)
        except ValueError as e:
            print(f"  [SKIP] {e}"); continue

        all_summaries[ds_name] = {}

        for n_train in sorted(args.n_train):
            if n_train > len(df):
                print(f"  [SKIP] n_train={n_train} > dataset size {len(df)}")
                continue
            print(f"\n  ── n_train={n_train}  ({args.n_seeds} seeds) ──")
            runs    = collect_seed_results(df, n_train, args.n_seeds, args.test_frac)
            if runs is None:
                print(f"  [SKIP] n_train={n_train} > training pool "
                      f"(dataset {len(df)} minus held-out test rows)")
                continue
            summary = summarise(runs)
            all_summaries[ds_name][n_train] = summary

            for method, s in summary.items():
                m_mean, r2_mean = s['mean'][0], s['mean'][1]
                m_std,  r2_std  = s['std'][0],  s['std'][1]
                print(f"    {method:<28}  MAPE {m_mean:6.2f}% ± {m_std:5.2f}%  "
                      f"R² {r2_mean:.4f} ± {r2_std:.4f}")
                csv_rows.append({
                    'dataset': ds_name, 'n_train': n_train, 'method': method,
                    'mape_mean': round(m_mean, 3), 'mape_std': round(m_std, 3),
                    'mape_min':  round(s['min'][0], 3), 'mape_max': round(s['max'][0], 3),
                    'r2_mean':   round(r2_mean, 4),     'r2_std':   round(r2_std, 4),
                    'rmse_mean': round(s['mean'][2], 4), 'rmse_std': round(s['std'][2], 4),
                    'n_seeds':   args.n_seeds,
                })

    if csv_rows:
        csv_path = os.path.join(args.output_dir, 'variance_results.csv')
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"\n  Saved CSV → {csv_path}")

    print_variance_table(all_summaries)

    if not args.no_plots and all_summaries:
        import matplotlib.ticker
        matplotlib.ticker  # already imported above; ensure available in plot fns
        print(f"\n  Generating plots → {args.output_dir}/")
        plot_boxplots(all_summaries, args.output_dir)
        plot_errorbars(all_summaries, args.output_dir)
        plot_std_heatmap(all_summaries, args.output_dir)

    print(f"\n{'━'*70}")
    print("  Done.")


if __name__ == '__main__':
    import matplotlib.ticker  # needed inside plot helpers
    main()
