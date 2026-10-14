#!/usr/bin/env python3
"""Ranking vs. MAPE — selection robustness under high absolute error

    python ranking_vs_mape.py
    python ranking_vs_mape.py --n-train 50 --seed 0
    python ranking_vs_mape.py --datasets "Mistral-7B (NVIDIA A100)"
    python ranking_vs_mape.py --min-mape 25        # threshold for "high MAPE"
"""

import argparse, glob, os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from compare_cross_model import (
    DATASETS, prepare_df, fit_unified, unified_predict, holdout_scenarios,
)
from dataset_args import resolve_datasets, DATASETS_HELP


def discover_datasets(results_dir, pattern='profiling_metrics.csv'):
    """Recursively find every metrics CSV under ``results_dir``.

    Returns an ordered ``{name: path}`` dict. The name is the CSV's directory
    path relative to ``results_dir`` (e.g. ``run_text_only_llava-1.5-7b-hf`` or
    ``run_text_video_llava-1.5-7b-hf/res240p``), so nested per-resolution runs
    stay distinct.
    """
    found = {}
    for path in sorted(glob.glob(os.path.join(results_dir, '**', pattern),
                                 recursive=True)):
        rel = os.path.relpath(os.path.dirname(path), results_dir)
        name = rel if rel != '.' else os.path.splitext(os.path.basename(path))[0]
        found[name] = path
    return found


# ════════════════════════════════════════════════════════════
#  METRICS
# ════════════════════════════════════════════════════════════

def pairwise_accuracy(true_e, pred_e):
    """Fraction of config pairs whose true energy order is preserved."""
    n = len(true_e); correct = 0.0; total = 0
    for i in range(n):
        for j in range(i + 1, n):
            dt = true_e[i] - true_e[j]
            if abs(dt) < 1e-9:
                continue
            total += 1
            dp = pred_e[i] - pred_e[j]
            if dt * dp > 0:
                correct += 1
            elif abs(dp) < 1e-9:
                correct += 0.5
    return correct / total if total else np.nan


def scenario_rows(df, true_e, pred_e):
    """Per-scenario (max_tokens, batch_size) ranking/selection metrics.

    Returns a list of dicts, each also carrying the per-config table so we can
    print a concrete example.
    """
    work = df.copy()
    work['_true'] = true_e
    work['_pred'] = pred_e
    rows = []
    for (tok, bs), g in work.groupby(['max_tokens', 'batch_size']):
        cm = (g.groupby(['tp_degree', 'pp_degree'])
                .agg(true_e=('_true', 'mean'), pred_e=('_pred', 'mean'))
                .reset_index())
        if len(cm) < 2:
            continue
        t = cm['true_e'].values.astype(float)
        p = cm['pred_e'].values.astype(float)

        mape = float(np.mean(np.abs(p - t) / np.maximum(t, 1e-9)) * 100)
        pa = pairwise_accuracy(t, p)
        rho = (spearmanr(t, p)[0]
               if np.std(t) > 1e-9 and np.std(p) > 1e-9 else np.nan)
        bi = int(np.argmin(t))           # true-optimal config index
        pi = int(np.argmin(p))           # selected config index
        regret = float((t[pi] - t[bi]) / t[bi] * 100) if t[bi] > 0 else 0.0
        top1 = (bi == pi)

        cm = cm.assign(true_rank=cm['true_e'].rank().astype(int),
                       pred_rank=cm['pred_e'].rank().astype(int))
        rows.append({
            'max_tokens': int(tok), 'batch_size': int(bs),
            'n_configs': len(cm), 'mape': mape, 'pairwise_acc': pa,
            'spearman': rho, 'top1': top1, 'regret_pct': regret,
            'sel_idx': pi, 'opt_idx': bi, 'table': cm,
        })
    return rows


# ════════════════════════════════════════════════════════════
#  CONCRETE EXAMPLE PRINTING
# ════════════════════════════════════════════════════════════

def print_example(ds_name, row):
    cm = row['table']
    print(f"\n  ┌─ CONCRETE EXAMPLE — {ds_name}")
    print(f"  │  Scenario: max_tokens={row['max_tokens']}, batch_size={row['batch_size']}, "
          f"{row['n_configs']} candidate configs")
    print(f"  │  Scenario MAPE = {row['mape']:.1f}%  (high absolute error)")
    print(f"  │  Pairwise ranking acc = {row['pairwise_acc']*100:.0f}%   "
          f"Spearman = {row['spearman']:.2f}")
    print(f"  │")
    print(f"  │  {'TP':>3} {'PP':>3} {'true_E(J)':>11} {'pred_E(J)':>11} "
          f"{'|err|%':>7} {'true_rk':>8} {'pred_rk':>8}")
    print(f"  │  {'-'*54}")
    for _, r in cm.sort_values('true_e').iterrows():
        err = abs(r['pred_e'] - r['true_e']) / max(r['true_e'], 1e-9) * 100
        mark_t = ' *OPT' if int(r['true_rank']) == 1 else ''
        mark_p = ' <pick' if int(r['pred_rank']) == 1 else ''
        print(f"  │  {int(r['tp_degree']):>3} {int(r['pp_degree']):>3} "
              f"{r['true_e']:>11.1f} {r['pred_e']:>11.1f} {err:>7.1f} "
              f"{int(r['true_rank']):>8}{mark_t:>0} {int(r['pred_rank']):>5}{mark_p:>0}")
    opt = cm.iloc[row['opt_idx']]; sel = cm.iloc[row['sel_idx']]
    print(f"  │")
    print(f"  │  True-optimal : TP={int(opt['tp_degree'])}, PP={int(opt['pp_degree'])}  "
          f"({opt['true_e']:.1f} J)")
    print(f"  │  EnergyLens picks: TP={int(sel['tp_degree'])}, PP={int(sel['pp_degree'])}  "
          f"({sel['true_e']:.1f} J)")
    verdict = "CORRECT selection, regret 0.0%" if row['top1'] else \
              f"different config, regret {row['regret_pct']:.1f}%"
    print(f"  │  → Despite {row['mape']:.1f}% MAPE, EnergyLens makes the {verdict}.")
    print(f"  └─")


# ════════════════════════════════════════════════════════════
#  AGGREGATE PLOT
# ════════════════════════════════════════════════════════════

def plot_ranking_vs_mape(all_rows, output_dir):
    if not all_rows:
        return
    mape = np.array([r['mape'] for r in all_rows])
    top1 = np.array([1.0 if r['top1'] else 0.0 for r in all_rows])
    regret = np.array([r['regret_pct'] for r in all_rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (a) scatter: MAPE vs regret, colored by correct selection
    ok = top1 == 1.0
    ax1.scatter(mape[ok], regret[ok], c='#2a9d8f', s=28, alpha=0.7,
                label='Correct optimal pick (regret 0)')
    ax1.scatter(mape[~ok], regret[~ok], c='#e76f51', s=28, alpha=0.8,
                label='Wrong pick')
    ax1.set_xlabel('Scenario MAPE (%)')
    ax1.set_ylabel('Selection regret (%)')
    ax1.set_title('High MAPE does not imply wrong selection')
    ax1.legend(fontsize=9); ax1.grid(ls='--', alpha=0.3)

    # (b) Top-1 selection accuracy within MAPE buckets
    bins = [0, 10, 20, 30, 50, 1e9]
    labels = ['0–10', '10–20', '20–30', '30–50', '50+']
    idx = np.digitize(mape, bins) - 1
    acc, ns = [], []
    for b in range(len(labels)):
        sel = idx == b
        acc.append(top1[sel].mean() * 100 if sel.any() else np.nan)
        ns.append(int(sel.sum()))
    bars = ax2.bar(range(len(labels)), acc, color='#457b9d',
                   edgecolor='black', linewidth=0.5)
    for bar, a, n in zip(bars, acc, ns):
        if not np.isnan(a):
            ax2.annotate(f'{a:.0f}%\n(n={n})',
                         xy=(bar.get_x() + bar.get_width() / 2, a),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', fontsize=9)
    ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels)
    ax2.set_xlabel('Scenario MAPE bucket (%)')
    ax2.set_ylabel('Top-1 selection accuracy (%)')
    ax2.set_ylim(0, 109)
    ax2.axhline(100, color='gray', ls='--', alpha=0.4)
    ax2.set_title('Selection accuracy stays high as MAPE grows')
    ax2.grid(axis='y', ls='--', alpha=0.3)

    fig.tight_layout()
    out = os.path.join(output_dir, 'ranking_vs_mape.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved plot → {out}")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='Ranking vs MAPE selection-robustness test')
    ap.add_argument('--output-dir', default='results/ranking_vs_mape')
    ap.add_argument('--n-train', type=int, default=50,
                    help='Training samples for the EnergyLens fit (default 50)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--test-frac', type=float, default=0.3,
                    help='Fraction of scenarios (max_tokens x batch_size) held '
                         'out for scoring (default 0.3). 0 = old in-sample '
                         'behaviour (score every scenario, training rows included)')
    ap.add_argument('--split-seed', type=int, default=42,
                    help='Seed for the scenario train/test split')
    ap.add_argument('--min-mape', type=float, default=20.0,
                    help='MAPE threshold defining a "high-MAPE" example (default 20%%)')
    ap.add_argument('--datasets', nargs='+', default=None,
                    help=DATASETS_HELP + ' (default here: auto-discover under '
                         '--results-dir plus the DATASETS entries)')
    ap.add_argument('--results-dir', default='results',
                    help='Root folder scanned for profiling_metrics.csv files '
                         '(default: results)')
    ap.add_argument('--pattern', default='profiling_metrics.csv',
                    help='CSV filename to auto-discover under --results-dir')
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.datasets is None:
        # Default: auto-discover every metrics CSV under results/, then add any
        # named DATASETS not already covered by a discovered path.
        datasets = discover_datasets(args.results_dir, args.pattern)
        have = set(datasets.values())
        for k, v in DATASETS.items():
            if os.path.isfile(v) and v not in have:
                datasets[k] = v
    else:
        datasets = resolve_datasets(args.datasets, DATASETS, args.pattern)
        datasets = {k: v for k, v in datasets.items() if os.path.isfile(v)}
    if not datasets:
        print('[ERROR] No datasets available.'); return
    print(f"  Loaded {len(datasets)} dataset(s) from '{args.results_dir}'")

    all_rows = []
    example_pool = []   # (mape, ds_name, row) for high-MAPE correct picks
    csv_rows = []
    fit_times = []      # (ds_name, fit_seconds) — supports the "within minutes" claim

    for ds_name, path in datasets.items():
        print(f"\n{'━'*70}\n  {ds_name}")
        try:
            df = prepare_df(pd.read_csv(path), path)
        except ValueError as e:
            print(f"  [SKIP] {e}"); continue

        # Hold out whole scenarios; fit on a small sample (this is where
        # realistic, non-trivial MAPE comes from) drawn from the training
        # scenarios only, and score only the held-out scenarios.
        pool_df, test_df, n_test, n_scen = holdout_scenarios(
            df, args.test_frac, args.split_seed)
        if n_test:
            print(f"  Split: {n_test}/{n_scen} scenarios held out (seed {args.split_seed})")
        elif args.test_frac > 0:
            print(f"  [split] only {n_scen} scenarios — too few to hold out; "
                  f"IN-SAMPLE (not a fair generalization test)")
        n = len(pool_df)
        rng = np.random.RandomState(args.seed)
        train_df = pool_df if args.n_train >= n else \
            pool_df.iloc[rng.choice(n, args.n_train, replace=False)].reset_index(drop=True)
        t_fit = time.perf_counter()
        params = fit_unified(train_df)
        fit_secs = time.perf_counter() - t_fit
        fit_times.append((ds_name, fit_secs))
        true_e = test_df['measured_energy'].values.astype(float)
        pred_e = unified_predict(params, test_df)

        rows = scenario_rows(test_df, true_e, pred_e)
        if not rows:
            print("  [SKIP] no multi-config scenarios"); continue

        overall_mape = float(np.mean(np.abs(pred_e - true_e) / np.maximum(true_e, 1e-9)) * 100)
        top1 = np.mean([r['top1'] for r in rows]) * 100
        mreg = np.mean([r['regret_pct'] for r in rows])
        print(f"  Overall MAPE={overall_mape:.1f}%  |  scenarios={len(rows)}  "
              f"Top-1={top1:.1f}%  mean regret={mreg:.2f}%  "
              f"| fit time={fit_secs:.3f}s (n_train={min(args.n_train, n)})")

        for r in rows:
            r['dataset'] = ds_name
            all_rows.append(r)
            csv_rows.append({k: r[k] for k in
                             ('dataset', 'max_tokens', 'batch_size', 'n_configs',
                              'mape', 'pairwise_acc', 'spearman', 'top1', 'regret_pct')})
            if r['top1'] and r['mape'] >= args.min_mape:
                example_pool.append((r['mape'], ds_name, r))

    if not all_rows:
        print('\n[ERROR] No scenarios computed.'); return

    # ── The concrete example the reviewer asked for ──
    print(f"\n{'='*70}")
    print("  RANKING vs MAPE — CONCRETE EXAMPLE(S)")
    print(f"{'='*70}")
    if example_pool:
        example_pool.sort(key=lambda x: -x[0])   # highest MAPE first
        for mape, ds_name, row in example_pool[:3]:
            print_example(ds_name, row)
    else:
        print(f"  No scenario with MAPE ≥ {args.min_mape}% AND a correct pick was "
              f"found; lower --min-mape to inspect.")

    # ── Aggregate summary ──
    mape = np.array([r['mape'] for r in all_rows])
    top1 = np.array([r['top1'] for r in all_rows], dtype=float)
    print(f"\n{'='*70}")
    print("  AGGREGATE: selection accuracy within MAPE buckets")
    print(f"{'='*70}")
    for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 50), (50, 1e9)]:
        sel = (mape >= lo) & (mape < hi)
        if sel.any():
            label = f"{lo}–{hi if hi < 1e9 else '∞'}%"
            print(f"    MAPE {label:<8}  n={int(sel.sum()):>4}  "
                  f"Top-1 selection acc = {top1[sel].mean()*100:5.1f}%")
    hi_mask = mape >= args.min_mape
    if hi_mask.any():
        print(f"\n  Headline: across {int(hi_mask.sum())} scenarios with MAPE ≥ "
              f"{args.min_mape:.0f}%, EnergyLens still selects the true-optimal "
              f"config {top1[hi_mask].mean()*100:.1f}% of the time.")

    # ── Model-fit wall-clock (the "within minutes" claim) ──
    if fit_times:
        secs = np.array([t for _, t in fit_times])
        print(f"\n{'='*70}")
        print(f"  MODEL-FIT WALL-CLOCK  (n_train={args.n_train}, {len(secs)} datasets)")
        print(f"{'='*70}")
        for ds_name, t in fit_times:
            print(f"    {ds_name:<40} {t:>8.3f} s")
        print(f"  {'-'*54}")
        print(f"    {'mean ± std':<40} {secs.mean():>8.3f} ± {secs.std():.3f} s")
        print(f"    {'min – max':<40} {secs.min():>8.3f} – {secs.max():.3f} s")
        pd.DataFrame(fit_times, columns=['dataset', 'fit_seconds']).to_csv(
            os.path.join(args.output_dir, 'fit_times.csv'), index=False)

    pd.DataFrame(csv_rows).to_csv(
        os.path.join(args.output_dir, 'ranking_vs_mape.csv'), index=False)
    print(f"\n  Saved per-scenario CSV → {os.path.join(args.output_dir, 'ranking_vs_mape.csv')}")
    plot_ranking_vs_mape(all_rows, args.output_dir)
    print(f"\n{'━'*70}\n  Done.")


if __name__ == '__main__':
    main()
