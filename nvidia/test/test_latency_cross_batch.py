#!/usr/bin/env python3
"""
Tests D, E & F: EnergyLens vs MaverIQ
=====================================

TEST D — Latency vs Energy Optimum Divergence:
  For each scenario, identify the latency-optimal and energy-optimal (TP,PP)
  config. Report how often they diverge, and the energy penalty of using the
  latency-optimal config. This directly challenges MaverIQ's core assumption
  that optimising latency also optimises energy.

TEST E — Cross-Batch-Size Extrapolation:
  Train on small batch sizes (BS ≤ threshold), predict on large batch sizes.
  EnergyLens uses BS^β (sublinear), while MaverIQ uses linear BS scaling
  (Eq. 6 in their paper: L(BS) = L(1) + (BS-1)×O_BS). Tests which
  functional form extrapolates better to unseen batch sizes.

TEST F — Output-Length Extrapolation:
  Same, on max_tokens: train on the short limits, predict the long ones.
  MaverIQ profiles output lengths 10 and 20 only and extrapolates
  TTFT + out x TPOT (their latency_estimator.py).

Usage:
  python test_compare_maverIQ.py
  python test_compare_maverIQ.py --output-dir more_tests_results
"""

DATASETS = {
    "Mistral-7B":              "./results/run_text_only_Mistral-7B-Instruct-v0.3/profiling_metrics.csv",
    "NVIDIA-Nemotron-V2-9B":   "./results/run_text_only_NVIDIA-Nemotron-Nano-9B-v2/profiling_metrics.csv",
    "Qwen1.5-MoE-2.7B":       "./results/run_text_only_Qwen1.5-MoE-A2.7B/profiling_metrics.csv",
    "Qwen2-VL-7B-text":       "./results/run_text_only_Qwen2-VL-7B-Instruct/profiling_metrics.csv",
    "Qwen2-VL-7B-image-chat": "./results/run_text_image_Qwen2-VL-7B-Instruct/profiling_metrics.csv",
    "Qwen2-VL-7B-video-chat": "./results/run_text_video_Qwen2-VL-7B-Instruct/combined_metrics.csv",
    "Qwen2-VL-7B-video-only": "./results/run_video_only_Qwen2-VL-7B-Instruct/profiling_metrics.csv",
    "LLava-1.5-7B-video-chat": "./results/run_text_video_llava-1.5-7b-hf/combined_metrics.csv",
    "LLava-1.5-7B-image-chat": "./results/run_text_image_llava-1.5-7b-hf/profiling_metrics.csv",
    "Qwen2.5-14B-fp16 (zeus)": "./results_zeus/run_text_only_Qwen2.5-14B-Instruct_fp16/res_native/profiling_metrics.csv",
    "Qwen2.5-14B-text":        "./results/run_text_only_Qwen2.5-14B-Instruct/res_native/profiling_metrics.csv",
    "Qwen2.5-14B-bnb4bit":     "./results/run_text_only_Qwen2.5-14B-Instruct_fp16-bnb4bit/res_native/profiling_metrics.csv",
}

import argparse, sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_args import add_datasets_arg, resolve_datasets
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize as opt
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')


# ════════════════════════════════════════════════════════════
#  DATA LOADING AND MODELS
# ════════════════════════════════════════════════════════════
# Shared with pairwise_ranking.py, so this test uses the same column mapping
# and the same EnergyLens / MaverIQ fits (batch size capped at max_num_seqs,
# longest output as decode length) as the ranking results.
from pairwise_ranking import (prepare_df, unified_predict, fit_unified,
                              fit_maveriq, maveriq_energy)


# ════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ════════════════════════════════════════════════════════════

def pairwise_accuracy(true_e, pred_e):
    n = len(true_e); correct = 0; total = 0
    for i in range(n):
        for j in range(i+1, n):
            d_true = true_e[i] - true_e[j]
            if abs(d_true) < 1e-9: continue
            total += 1
            d_pred = pred_e[i] - pred_e[j]
            if d_true * d_pred > 0: correct += 1
            elif abs(d_pred) < 1e-9: correct += 0.5
    return correct / total if total > 0 else np.nan

def scenario_metrics(df, pred_dict):
    """Compute per-scenario pairwise metrics."""
    results = []
    for (tok, bs), group in df.groupby(['max_tokens', 'batch_size']):
        config_means = group.groupby(['tp_degree','pp_degree']).agg(
            true_energy=('measured_energy','mean'),
            true_latency=('measured_latency','mean')).reset_index()
        if len(config_means) < 2: continue
        cfgs = []
        for _, row in config_means.iterrows():
            tp_v, pp_v = row['tp_degree'], row['pp_degree']
            mask = (group['tp_degree']==tp_v) & (group['pp_degree']==pp_v)
            idx = group[mask].index
            entry = {'tp': tp_v, 'pp': pp_v,
                     'true_energy': row['true_energy'],
                     'true_latency': row['true_latency']}
            for name, pred_arr in pred_dict.items():
                entry[name] = np.mean(pred_arr[idx])
            cfgs.append(entry)
        if len(cfgs) < 2: continue
        true_e = np.array([c['true_energy'] for c in cfgs])
        true_l = np.array([c['true_latency'] for c in cfgs])
        for name in pred_dict:
            pred_arr = np.array([c[name] for c in cfgs])
            pa = pairwise_accuracy(true_e, pred_arr)
            rho = spearmanr(true_e, pred_arr)[0] if np.std(true_e)>1e-9 and np.std(pred_arr)>1e-9 else np.nan
            bi = np.argmin(true_e); pi = np.argmin(pred_arr)
            oracle_e = true_e[bi]; sel_e = true_e[pi]
            regret = (sel_e - oracle_e) / oracle_e * 100 if oracle_e > 0 else 0
            results.append({'method': name, 'tok': tok, 'bs': bs,
                            'pairwise_acc': pa, 'spearman': rho,
                            'top1': bi==pi, 'regret_pct': regret,
                            'n_configs': len(cfgs),
                            'true_energy_arr': true_e,
                            'true_latency_arr': true_l,
                            'config_labels': [(c['tp'],c['pp']) for c in cfgs]})
    return results


# ════════════════════════════════════════════════════════════
#  TEST D — Latency vs Energy Optimum Divergence
# ════════════════════════════════════════════════════════════

def test_d_latency_energy_divergence(df, dataset_name):
    """For each scenario, compare the latency-optimal and energy-optimal config.
    Quantifies the energy penalty of using MaverIQ's latency-based approach."""
    print(f"\n  ── TEST D: Latency vs Energy Optimum Divergence ──")

    rows = []
    for (tok, bs), group in df.groupby(['max_tokens', 'batch_size']):
        config_means = group.groupby(['tp_degree','pp_degree']).agg(
            mean_energy=('measured_energy','mean'),
            mean_latency=('measured_latency','mean')).reset_index()
        if len(config_means) < 2:
            continue

        energies = config_means['mean_energy'].values
        latencies = config_means['mean_latency'].values
        configs = list(zip(config_means['tp_degree'], config_means['pp_degree']))

        energy_best_idx = np.argmin(energies)
        latency_best_idx = np.argmin(latencies)
        same_optimum = (energy_best_idx == latency_best_idx)

        oracle_energy = energies[energy_best_idx]
        latency_pick_energy = energies[latency_best_idx]
        energy_penalty_pct = (latency_pick_energy - oracle_energy) / oracle_energy * 100 if oracle_energy > 0 else 0

        oracle_latency = latencies[latency_best_idx]
        energy_pick_latency = latencies[energy_best_idx]
        latency_penalty_pct = (energy_pick_latency - oracle_latency) / oracle_latency * 100 if oracle_latency > 0 else 0

        rho = spearmanr(energies, latencies)[0] if np.std(energies) > 1e-9 and np.std(latencies) > 1e-9 else np.nan

        rows.append({
            'dataset': dataset_name, 'max_tokens': tok, 'batch_size': bs,
            'n_configs': len(config_means),
            'energy_best_config': str(configs[energy_best_idx]),
            'latency_best_config': str(configs[latency_best_idx]),
            'same_optimum': same_optimum,
            'energy_penalty_of_latency_pick_pct': round(energy_penalty_pct, 2),
            'latency_penalty_of_energy_pick_pct': round(latency_penalty_pct, 2),
            'energy_latency_spearman': round(rho, 4) if not np.isnan(rho) else np.nan,
            'oracle_energy': round(oracle_energy, 4),
            'latency_pick_energy': round(latency_pick_energy, 4),
        })

    if not rows:
        print(f"    No scenarios with ≥2 configs")
        return rows

    rdf = pd.DataFrame(rows)
    n_total = len(rdf); n_same = rdf['same_optimum'].sum(); n_diff = n_total - n_same
    mean_penalty = rdf['energy_penalty_of_latency_pick_pct'].mean()
    max_penalty = rdf['energy_penalty_of_latency_pick_pct'].max()
    mean_rho = rdf['energy_latency_spearman'].mean()
    pct_gt5 = (rdf['energy_penalty_of_latency_pick_pct'] > 5).mean() * 100
    pct_gt10 = (rdf['energy_penalty_of_latency_pick_pct'] > 10).mean() * 100
    pct_gt25 = (rdf['energy_penalty_of_latency_pick_pct'] > 25).mean() * 100
    total_oracle_e = rdf['oracle_energy'].sum()
    total_latpick_e = rdf['latency_pick_energy'].sum()
    total_overhead = (total_latpick_e - total_oracle_e) / total_oracle_e * 100

    print(f"    Scenarios: {n_total} | Same optimum: {n_same} ({n_same/n_total*100:.1f}%) | Different: {n_diff} ({n_diff/n_total*100:.1f}%)")
    print(f"    Energy penalty of latency-optimal pick:")
    print(f"      Mean: {mean_penalty:.2f}%  Max: {max_penalty:.1f}%  >5%: {pct_gt5:.1f}%  >10%: {pct_gt10:.1f}%  >25%: {pct_gt25:.1f}%")
    print(f"    Total energy: Oracle={total_oracle_e:.2f}J  Latency-pick={total_latpick_e:.2f}J  Overhead={total_overhead:.2f}%")
    print(f"    Energy↔Latency Spearman ρ (mean across scenarios): {mean_rho:.3f}")

    return rows


# ════════════════════════════════════════════════════════════
#  TEST E — Cross-Batch-Size Extrapolation
# ════════════════════════════════════════════════════════════

EXTRAPOLATION = {   # column -> (test letter, title, plot axis label)
    'batch_size': ('E', 'Cross-Batch-Size Extrapolation', 'Batch Sizes'),
    'max_tokens': ('F', 'Output-Length Extrapolation', 'Output Lengths'),
}


def extrapolation_test(df, dataset_name, col='batch_size'):
    """Train on the small values of `col`, test ranking accuracy on the large ones.

    Test E (batch_size): EnergyLens' BS^beta vs MaverIQ's linear batch overhead.
    Test F (max_tokens): MaverIQ profiles output lengths 10 and 20 and extrapolates
    TTFT + out x TPOT to longer outputs; here both models are fit on the short
    max_tokens limits only and scored on the long ones."""
    letter, title, _ = EXTRAPOLATION[col]
    print(f"\n  ── TEST {letter}: {title} ──")

    bs_values = sorted(df[col].unique())
    if len(bs_values) < 4:
        print(f"    Skipped (only {len(bs_values)} unique values of {col})")
        return []

    # Find a split point: roughly bottom 40% of the values for training (at least 2)
    split_idx = max(2, len(bs_values) * 2 // 5)
    bs_train_max = bs_values[split_idx - 1]
    bs_train_set = set(bs_values[:split_idx])
    bs_test_set = set(bs_values[split_idx:])

    train_mask = df[col].isin(bs_train_set)
    test_mask = df[col].isin(bs_test_set)
    train_df = df[train_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    if len(train_df) < 20 or len(test_df) < 10:
        print(f"    Skipped (too few rows: train={len(train_df)}, test={len(test_df)})")
        return []

    print(f"    Train {col}: {sorted(bs_train_set)} ({len(train_df)} rows)")
    print(f"    Test {col}:  {sorted(bs_test_set)} ({len(test_df)} rows)")

    # Fit on train, predict on test
    try:
        uni_params = fit_unified(train_df, restarts=10)
        uni_pred_test = unified_predict(uni_params, test_df)
    except:
        uni_pred_test = np.full(len(test_df), np.nan)

    try:
        _, mav_fn = fit_maveriq(train_df)
        mav_pred_test = maveriq_energy(mav_fn, test_df)
    except:
        mav_pred_test = np.full(len(test_df), np.nan)

    # Also fit on ALL data for in-sample reference
    try:
        uni_params_full = fit_unified(df, restarts=10)
        uni_pred_full_test = unified_predict(uni_params_full, test_df)
    except:
        uni_pred_full_test = np.full(len(test_df), np.nan)
    try:
        _, mav_fn_full = fit_maveriq(df)
        mav_pred_full_test = maveriq_energy(mav_fn_full, test_df)
    except:
        mav_pred_full_test = np.full(len(test_df), np.nan)

    # Compute MAPE on test set
    true_test = test_df['measured_energy'].values
    methods = {
        'EnergyLens_extrap': uni_pred_test,
        'MaverIQ_extrap': mav_pred_test,
        'EnergyLens_full': uni_pred_full_test,
        'MaverIQ_full': mav_pred_full_test,
    }

    rows = []
    print(f"\n    {'Method':<22} {'MAPE':>7} {'PairAcc':>8} {'Spearman':>9} {'Top1%':>7} {'Regret':>8}")
    print(f"    {'─'*65}")

    for name, pred in methods.items():
        valid = ~np.isnan(pred)
        if valid.sum() < 10: continue
        mape = np.mean(np.abs(pred[valid] - true_test[valid]) / np.maximum(true_test[valid], 1e-6)) * 100
        metrics = scenario_metrics(test_df[valid].reset_index(drop=True),
                                    {name: pred[valid]})
        m_rows = [r for r in metrics if r['method'] == name]
        if m_rows:
            pa = np.mean([r['pairwise_acc'] for r in m_rows]) * 100
            sp = np.nanmean([r['spearman'] for r in m_rows])
            t1 = np.mean([r['top1'] for r in m_rows]) * 100
            reg = np.mean([r['regret_pct'] for r in m_rows])
            ns = len(m_rows)
        else:
            pa = sp = t1 = reg = np.nan; ns = 0

        label = 'extrapolated' if 'extrap' in name else 'in-sample'
        print(f"    {name:<22} {mape:>6.1f}% {pa:>7.1f}% {sp:>8.3f} {t1:>6.1f}% {reg:>7.2f}%")
        rows.append({'dataset': dataset_name, 'method': name, 'eval_type': label,
                     'axis': col, 'train_max': int(bs_train_max),
                     'mape': round(mape, 2), 'pairwise_acc': round(pa, 2),
                     'spearman': round(sp, 4), 'top1_acc': round(t1, 2),
                     'mean_regret': round(reg, 2), 'n_scenarios': ns,
                     'n_train': len(train_df), 'n_test': len(test_df)})

    return rows


# ════════════════════════════════════════════════════════════
#  PLOTTING
# ════════════════════════════════════════════════════════════

def plot_test_d(all_d_rows, output_dir):
    """Bar chart: energy penalty of latency-optimal pick per dataset."""
    plt.ioff()
    dfd = pd.DataFrame(all_d_rows)
    if dfd.empty: return

    agg = dfd.groupby('dataset').agg(
        mean_penalty=('energy_penalty_of_latency_pick_pct', 'mean'),
        pct_diverge=('same_optimum', lambda x: (1 - x.mean()) * 100),
        mean_rho=('energy_latency_spearman', 'mean'),
    ).reset_index().sort_values('mean_penalty', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(agg))

    ax = axes[0]
    bars = ax.bar(x, agg['mean_penalty'], color='#e76f51', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(agg['dataset'], rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Energy Penalty (%)', fontsize=11)
    ax.set_title('Energy Penalty of Choosing Latency-Optimal Config\n(MaverIQ\'s implicit assumption)', fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    for bar, v in zip(bars, agg['mean_penalty']):
        ax.annotate(f'{v:.1f}%', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8, fontweight='bold')

    ax = axes[1]
    bars = ax.bar(x, agg['pct_diverge'], color='#264653', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(agg['dataset'], rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Scenarios with Different Optima (%)', fontsize=11)
    ax.set_title('How Often Latency-Optimal ≠ Energy-Optimal Config', fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    for bar, v in zip(bars, agg['pct_diverge']):
        ax.annotate(f'{v:.0f}%', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8, fontweight='bold')

    fig.tight_layout()
    out = os.path.join(output_dir, 'test_d_latency_energy_divergence.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved → {out}")


def plot_test_e(all_e_rows, output_dir, col='batch_size', fname='test_e_batch_extrapolation.png'):
    """Grouped bar chart: extrapolated vs in-sample MAPE and pairwise accuracy."""
    unseen = EXTRAPOLATION[col][2]
    plt.ioff()
    dfe = pd.DataFrame(all_e_rows)
    if dfe.empty: return

    datasets = sorted(dfe['dataset'].unique())
    methods_order = ['EnergyLens_extrap', 'MaverIQ_extrap', 'EnergyLens_full', 'MaverIQ_full']
    colors = {'EnergyLens_extrap': '#2a9d8f', 'MaverIQ_extrap': '#e76f51',
              'EnergyLens_full': '#76c7b7', 'MaverIQ_full': '#f4a582'}
    labels = {'EnergyLens_extrap': 'EnergyLens (extrapolated)',
              'MaverIQ_extrap': 'MaverIQ (extrapolated)',
              'EnergyLens_full': 'EnergyLens (in-sample)',
              'MaverIQ_full': 'MaverIQ (in-sample)'}

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(len(datasets))
    width = 0.2

    for ax, metric, title, ylabel in [
        (axes[0], 'pairwise_acc', f'Pairwise Ranking Accuracy on Unseen {unseen}', 'Accuracy (%)'),
        (axes[1], 'mape', f'MAPE on Unseen {unseen}', 'MAPE (%)'),
    ]:
        for mi, method in enumerate(methods_order):
            vals = []
            for ds in datasets:
                row = dfe[(dfe['dataset']==ds) & (dfe['method']==method)]
                vals.append(row[metric].values[0] if len(row) > 0 else 0)
            bars = ax.bar(x + (mi-1.5)*width, vals, width,
                         label=labels.get(method, method),
                         color=colors.get(method, 'gray'), edgecolor='black', linewidth=0.5)
            for bar, v in zip(bars, vals):
                if v > 0:
                    fmt = f'{v:.1f}%' if metric == 'pairwise_acc' else f'{v:.0f}%'
                    ax.annotate(fmt, xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                               xytext=(0, 3), textcoords='offset points', ha='center', fontsize=6, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=7, ncol=2); ax.grid(axis='y', linestyle='--', alpha=0.3)
        if metric == 'pairwise_acc':
            ax.axhline(50, color='gray', linestyle='--', alpha=0.4)

    fig.tight_layout()
    out = os.path.join(output_dir, fname)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out}")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='Three More Tests')
    ap.add_argument('--output-dir', type=str, default='results_figures')
    add_datasets_arg(ap)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    datasets  = resolve_datasets(args.datasets, DATASETS)
    available = {k: v for k, v in datasets.items() if os.path.isfile(v)}
    if not available:
        print("No datasets found."); sys.exit(1)
    missing = {k: v for k, v in datasets.items() if not os.path.isfile(v)}
    if missing:
        print(f"Skipping {len(missing)} missing: {list(missing.keys())}")

    all_d, all_e, all_f = [], [], []

    for ds_name, ds_path in available.items():
        print(f"\n{'='*70}")
        print(f"  {ds_name}")
        print(f"{'='*70}")
        try:
            raw = pd.read_csv(ds_path)
            df = prepare_df(raw, ds_path)
            print(f"  {len(df)} rows | TP={sorted(df['tp_degree'].unique())} "
                  f"PP={sorted(df['pp_degree'].unique())} | "
                  f"BS: {len(df['batch_size'].unique())} unique | "
                  f"MaxTok={sorted(df['max_tokens'].unique())}")

            # TEST D
            d_rows = test_d_latency_energy_divergence(df, ds_name)
            all_d.extend(d_rows)

            # TEST E
            e_rows = extrapolation_test(df, ds_name, 'batch_size')
            all_e.extend(e_rows)

            # TEST F
            all_f.extend(extrapolation_test(df, ds_name, 'max_tokens'))

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    # ── TEST D Summary ──
    if all_d:
        print(f"\n{'='*90}")
        print(f"  TEST D SUMMARY: Latency vs Energy Optimum Divergence")
        print(f"{'='*90}")
        dfd = pd.DataFrame(all_d)
        agg = dfd.groupby('dataset').agg(
            n_scenarios=('same_optimum', 'count'),
            pct_same=('same_optimum', lambda x: x.mean()*100),
            mean_energy_penalty=('energy_penalty_of_latency_pick_pct', 'mean'),
            max_energy_penalty=('energy_penalty_of_latency_pick_pct', 'max'),
            mean_rho=('energy_latency_spearman', 'mean'),
        ).reset_index()
        print(f"\n  {'Dataset':<25} {'#Scen':>6} {'Same%':>7} {'MeanPen':>8} {'MaxPen':>8} {'Lat↔E ρ':>8}")
        print(f"  {'─'*65}")
        for _, r in agg.iterrows():
            print(f"  {r['dataset']:<25} {r['n_scenarios']:>6} {r['pct_same']:>6.1f}% "
                  f"{r['mean_energy_penalty']:>7.2f}% {r['max_energy_penalty']:>7.1f}% {r['mean_rho']:>7.3f}")
        n_total = len(dfd); n_diff = (~dfd['same_optimum']).sum()
        print(f"\n  OVERALL: {n_total} scenarios, {n_diff} ({n_diff/n_total*100:.1f}%) have different latency/energy optima")
        print(f"  Mean energy penalty of latency-proxy selection: {dfd['energy_penalty_of_latency_pick_pct'].mean():.2f}%")
        print(f"  Max energy penalty: {dfd['energy_penalty_of_latency_pick_pct'].max():.1f}%")

    # ── TEST E / F Summaries ──
    for col, rows in (('batch_size', all_e), ('max_tokens', all_f)):
        if not rows:
            continue
        letter, title, _ = EXTRAPOLATION[col]
        print(f"\n{'='*90}")
        print(f"  TEST {letter} SUMMARY: {title}")
        print(f"{'='*90}")
        print(f"\n  {'Dataset':<25} {'Method':<22} {'Type':<14} {'MAPE':>7} {'PairAcc':>8} {'Spearman':>9} {'Top1':>7} {'Regret':>8}")
        print(f"  {'─'*103}")
        for r in sorted(rows, key=lambda x: (x['dataset'], x['method'])):
            print(f"  {r['dataset']:<25} {r['method']:<22} {r['eval_type']:<14} "
                  f"{r['mape']:>6.1f}% {r['pairwise_acc']:>7.1f}% {r['spearman']:>8.3f} {r['top1_acc']:>6.1f}% {r['mean_regret']:>7.2f}%")
        mean = pd.DataFrame(rows).groupby('method')[['mape', 'pairwise_acc', 'top1_acc', 'mean_regret']].mean()
        print(f"\n  Mean over datasets:\n{mean.round(2).to_string()}")

    # ── Save CSVs ──
    for name, rows in [('test_d_divergence', all_d),
                        ('test_e_batch_extrap', all_e),
                        ('test_f_output_extrap', all_f)]:
        if rows:
            path = os.path.join(args.output_dir, f'{name}.csv')
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"\n  Saved → {path}")

    # ── Plots ──
    plot_test_d(all_d, args.output_dir)
    plot_test_e(all_e, args.output_dir)
    plot_test_e(all_f, args.output_dir, 'max_tokens', 'test_f_output_extrapolation.png')

    print(f"\n  All results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()