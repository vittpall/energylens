    #!/usr/bin/env python3
"""
TEST C — No-Power Degradation: EnergyLens vs MaverIQ
=====================================================

MaverIQ uses latency × measured_power. In deployment, you don't have
per-config power measurements before profiling. This test replaces measured
power with a constant (mean power) and measures how MaverIQ degrades.
EnergyLens doesn't use power at all — it predicts energy directly.

Usage:
  python final_test.py
  python final_test.py --output-dir results
"""

DATASETS = {
    "Mistral-7B":              "./results/run_text_only_Mistral-7B-Instruct-v0.3/profiling_metrics.csv",
    "NVIDIA-Nemotron-V2-9B":   "./results/run_text_only_NVIDIA-Nemotron-Nano-9B-v2/profiling_metrics.csv",
    "Qwen1.5-MoE-2.7B":       "./results/run_text_only_Qwen1.5-MoE-A2.7B/profiling_metrics.csv",
    "Qwen2-VL-7B-text":       "./results/run_text_only_Qwen2-VL-7B-Instruct/profiling_metrics.csv",
    "Qwen2-VL-7B-image-chat": "./results/run_text_image_Qwen2-VL-7B-Instruct/profiling_metrics.csv",
    "Qwen2-VL-7B-video-chat": "./results/run_text_video_Qwen2-VL-7B-Instruct/combined_metrics.csv",
    "Qwen2-VL-7B-video-only": "./results/run_video_only_Qwen2-VL-7B-Instruct/profiling_metrics.csv",
    #"LLava-1.5-7B-video-chat": "./results/run_text_video_llava-1.5-7b-hf/combined_metrics.csv",
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

def maveriq_energy_const_power(lat_fn, df, const_power):
    # Same per-batch -> per-request correction as maveriq_energy.
    bs = np.maximum(df['batch_size'].values.astype(float), 1.0)
    return lat_fn(df) * const_power / bs

# ════════════════════════════════════════════════════════════
#  PAIRWISE METRICS (shared helper)
# ════════════════════════════════════════════════════════════

def pairwise_accuracy(true_e, pred_e):
    """Fraction of pairs ranked correctly."""
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

def scenario_metrics(df, true_all, pred_dict):
    """Compute per-scenario pairwise metrics. Returns list of dicts."""
    results = []
    for (tok, bs), group in df.groupby(['max_tokens', 'batch_size']):
        config_means = group.groupby(['tp_degree','pp_degree']).agg(
            true_energy=('measured_energy','mean')).reset_index()
        if len(config_means) < 2: continue
        cfgs = []
        for _, row in config_means.iterrows():
            tp_v, pp_v = row['tp_degree'], row['pp_degree']
            mask = (group['tp_degree']==tp_v) & (group['pp_degree']==pp_v)
            idx = group[mask].index
            entry = {'tp': tp_v, 'pp': pp_v, 'true_energy': row['true_energy']}
            for name, pred_arr in pred_dict.items():
                entry[name] = np.mean(pred_arr[idx])
            cfgs.append(entry)
        if len(cfgs) < 2: continue
        true_arr = np.array([c['true_energy'] for c in cfgs])
        for name in pred_dict:
            pred_arr = np.array([c[name] for c in cfgs])
            pa = pairwise_accuracy(true_arr, pred_arr)
            rho = spearmanr(true_arr, pred_arr)[0] if np.std(true_arr) > 1e-9 and np.std(pred_arr) > 1e-9 else np.nan
            bi = np.argmin(true_arr); pi = np.argmin(pred_arr)
            oracle_e = true_arr[bi]; sel_e = true_arr[pi]
            regret = (sel_e - oracle_e) / oracle_e * 100 if oracle_e > 0 else 0
            results.append({'method': name, 'tok': tok, 'bs': bs,
                            'pairwise_acc': pa, 'spearman': rho,
                            'top1': bi == pi, 'regret_pct': regret,
                            'n_configs': len(cfgs)})
    return results


# ════════════════════════════════════════════════════════════
#  TEST C — MaverIQ No-Power Degradation
# ════════════════════════════════════════════════════════════

def test_c_no_power(df, dataset_name, output_dir):
    """Compare MaverIQ with measured power vs constant power vs EnergyLens."""
    print(f"\n  ── TEST C: No-Power Degradation ──")

    n = len(df)
    true_e = df['measured_energy'].values
    mean_power = df['gpu_power_w'].mean()
    median_power = df['gpu_power_w'].median()

    # Fit all models
    uni_params = fit_unified(df, restarts=10)
    _, mav_fn = fit_maveriq(df)

    uni_pred = unified_predict(uni_params, df)
    mav_mean_pw  = maveriq_energy_const_power(mav_fn, df, mean_power)    # latency × mean_power

    methods = {
        'EnergyLens':             uni_pred,
        'MaverIQ_mean_P':         mav_mean_pw,
    }

    # MAPE
    print(f"    Mean GPU power: {mean_power:.1f}W  Median: {median_power:.1f}W")
    for name, pred in methods.items():
        mape = np.mean(np.abs(pred - true_e) / np.maximum(true_e, 1e-6)) * 100
        print(f"    {name:<25} MAPE: {mape:.1f}%")

    # Pairwise ranking
    metrics = scenario_metrics(df, true_e, methods)

    all_rows = []
    print(f"\n    {'Method':<25} {'PairAcc':>8} {'Spearman':>9} {'Top1%':>7} {'MeanReg':>8} {'#Scen':>6}")
    print(f"    {'─'*70}")
    for name in methods:
        rows = [r for r in metrics if r['method'] == name]
        if not rows: continue
        pa  = np.mean([r['pairwise_acc'] for r in rows]) * 100
        sp  = np.nanmean([r['spearman'] for r in rows])
        t1  = np.mean([r['top1'] for r in rows]) * 100
        reg = np.mean([r['regret_pct'] for r in rows])
        ns  = len(rows)
        print(f"    {name:<25} {pa:>7.1f}% {sp:>8.3f} {t1:>6.1f}% {reg:>7.2f}% {ns:>6}")
        mape = np.mean(np.abs(methods[name] - true_e) / np.maximum(true_e, 1e-6)) * 100
        all_rows.append({'dataset': dataset_name, 'method': name,
                         'pairwise_acc': pa, 'spearman': sp, 'top1_acc': t1,
                         'mean_regret': reg, 'mape': mape, 'n_scenarios': ns})
    return all_rows


# ════════════════════════════════════════════════════════════
#  PLOTTING
# ════════════════════════════════════════════════════════════

def plot_no_power(all_c_rows, output_dir):
    """Bar chart showing MaverIQ degradation without measured power."""
    plt.ioff()
    dfc = pd.DataFrame(all_c_rows)
    if dfc.empty: return

    datasets = sorted(dfc['dataset'].unique())
    
    # Updated to ONLY include EnergyLens and MaverIQ_mean_P
    methods = ['EnergyLens', 'MaverIQ_mean_P']
    colors_map = {'EnergyLens': '#2a9d8f', 'MaverIQ_mean_P': '#f4a582'}
    labels_map = {'EnergyLens': 'EnergyLens (no power needed)',
                  'MaverIQ_mean_P': 'MaverIQ (mean power)'}

    fig, ax = plt.subplots(figsize=(max(10, len(datasets)*2), 6))
    x = np.arange(len(datasets))
    
    # Adjust bar width and centering for just two bars
    width = 0.35 

    for mi, method in enumerate(methods):
        vals = []
        for ds in datasets:
            row = dfc[(dfc['dataset'] == ds) & (dfc['method'] == method)]
            vals.append(row['pairwise_acc'].values[0] if len(row) > 0 else 0)
            
        # Shift positions so the two bars are centered over the tick
        bars = ax.bar(x + (mi - 0.5) * width, vals, width,
                      label=labels_map.get(method, method),
                      color=colors_map.get(method, 'gray'), edgecolor='black', linewidth=0.5)
                      
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.annotate(f'{v:.1f}%', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points', ha='center', fontsize=13, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Pairwise Accuracy (%)', fontsize=11)
    ax.legend(fontsize=8, loc='lower left'); ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.4)

    fig.tight_layout()
    out = os.path.join(output_dir, 'test_c_no_power.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out}")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='Three-Test Evaluation')
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

    all_c = []

    for ds_name, ds_path in available.items():
        print(f"\n{'='*70}")
        print(f"  {ds_name}")
        print(f"{'='*70}")
        try:
            raw = pd.read_csv(ds_path)
            df = prepare_df(raw, ds_path)
            print(f"  {len(df)} rows | TP={sorted(df['tp_degree'].unique())} PP={sorted(df['pp_degree'].unique())} "
                  f"BS={len(df['batch_size'].unique())} unique | MaxTok={sorted(df['max_tokens'].unique())}")

            all_c.extend(test_c_no_power(df, ds_name, args.output_dir))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    # ── Save CSV ──
    if all_c:
        path = os.path.join(args.output_dir, 'test_c_no_power.csv')
        pd.DataFrame(all_c).to_csv(path, index=False)
        print(f"\n  Saved → {path}")

    # ── Summary Table ──
    if all_c:
        print(f"\n{'='*90}")
        print(f"  TEST C SUMMARY: No-Power Degradation")
        print(f"{'='*90}")
        print(f"\n  {'Dataset':<25} {'Method':<25} {'PairAcc':>8} {'Spearman':>9} {'Top1%':>7} {'MAPE':>7} {'Regret':>8}")
        print(f"  {'─'*95}")
        for r in all_c:
            print(f"  {r['dataset']:<25} {r['method']:<25} {r['pairwise_acc']:>7.1f}% {r['spearman']:>8.3f} "
                  f"{r['top1_acc']:>6.1f}% {r['mape']:>6.1f}% {r['mean_regret']:>7.2f}%")

    # ── Plot ──
    plot_no_power(all_c, args.output_dir)

    print(f"\n  All results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()