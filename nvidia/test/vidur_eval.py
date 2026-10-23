#!/usr/bin/env python3
"""Score the Vidur baseline against the other methods on the same held-out rows.

Vidur's latency for each grid batch comes from vidur_baseline.py (zero-shot: no
fitting on our rows). Its energy follows the paper's MaverIQ adaptation, the
predicted latency times the configuration's measured power divided by the batch
size; "Vidur, training power" uses each layout's median power over the training
rows instead, so that no measured value of the test row is used. Only the layouts Vidur's bundled profiles cover (TP 1 and 4) are scored, and
every method is restricted to the same rows. Per seed, scenarios ((max_tokens,
batch_size) points) are held out as in pairwise_ranking.py and the fitted methods
use 50 rows of the rest.

    python test/vidur_eval.py --grid results_zeus/run_text_only_Llama-3.1-8B-Instruct_fp16/res_native/profiling_metrics.csv
"""
import argparse, os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pairwise_ranking import (prepare_df, fit_unified, unified_predict, fit_maveriq, maveriq_energy,
                              fit_gb, _SKLEARN)
from compare_cross_model import holdout_scenarios, drop_invalid_rows
from vidur_baseline import keys as vidur_keys, SUPPORTED_TP


def ranking(df, preds):
    out = {m: {'pair': [], 'top1': [], 'regret': []} for m in preds}
    for _, g in df.groupby(['max_tokens', 'batch_size']):
        cfg = g.groupby(['tp_degree', 'pp_degree']).indices
        if len(cfg) < 2:
            continue
        y = np.array([g['measured_energy'].values[i].mean() for i in cfg.values()])
        for m, p in preds.items():
            pv = np.array([p[g.index.values[i]].mean() for i in cfg.values()])
            if np.isnan(pv).any():
                continue
            dy, dp = y[:, None] - y[None, :], pv[:, None] - pv[None, :]
            v = np.triu(np.abs(dy) > 1e-9, 1)
            out[m]['pair'].append(((dy * dp) > 0)[v].mean() if v.any() else 1.0)
            out[m]['top1'].append(pv.argmin() == y.argmin())
            out[m]['regret'].append((y[pv.argmin()] - y.min()) / y.min() * 100)
    return {m: {k: (100 * np.mean(v) if k != 'regret' else np.mean(v)) for k, v in r.items()} for m, r in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--grid', required=True)
    ap.add_argument('--vidur', default='results_figures/vidur/vidur_latency.csv')
    ap.add_argument('--n-train', type=int, default=50)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--out', default='results_figures/vidur')
    ap.add_argument('--tp', type=int, nargs='+', default=list(SUPPORTED_TP))
    args = ap.parse_args()

    raw = pd.read_csv(args.grid)
    raw = raw[pd.to_numeric(raw['energy'], errors='coerce') > 0].reset_index(drop=True)
    raw = drop_invalid_rows(raw, args.grid)        # before prepare_df, so the keys below stay aligned
    df = prepare_df(raw, args.grid)
    k = vidur_keys(args.grid)                      # same keys, same order as raw rows after the filter
    v = pd.read_csv(args.vidur)
    k = k.merge(v[['tp', 'pp', 'budget', 'bs', 'prefill', 'decode', 'vidur_latency']], how='left',
                on=['tp', 'pp', 'budget', 'bs', 'prefill', 'decode'])
    raw_keys = pd.DataFrame({'tp': raw['tensor_parallel_size'].astype(int), 'pp': raw['pipeline_parallel_size'].astype(int),
                             'budget': raw['max_num_batched_tokens'].astype(int), 'bs': raw['batch_size'].astype(int),
                             'prefill': (raw['input_length'] / raw['batch_size']).round().astype(int).clip(lower=1),
                             'decode': (raw['max_generated_tokens'] if 'max_generated_tokens' in raw else raw['max_tokens'])
                                       .fillna(raw['max_tokens']).astype(int).clip(lower=1)})
    lat = raw_keys.merge(k, how='left', on=list(raw_keys.columns))['vidur_latency'].to_numpy()
    if len(lat) != len(df):    # prepare_df dropped rows: realign through the measured values
        sys.exit(f'row mismatch {len(lat)} vs {len(df)}; re-run on a grid without invalid rows')
    df['vidur_latency'] = lat
    df = df[df['tp_degree'].isin(args.tp) & df['vidur_latency'].notna()].reset_index(drop=True)
    print(f'{len(df)} rows on layouts {sorted(set(zip(df.tp_degree, df.pp_degree)))}')
    vl = df['vidur_latency'].to_numpy()
    print(f"Vidur latency MAPE (all rows): {100 * np.mean(np.abs(vl - df.measured_latency) / df.measured_latency):.1f}%")

    rows = []
    for seed in range(args.seeds):
        train_pool, test, _, _ = holdout_scenarios(df, 0.3, seed + 42)
        tr = train_pool.sample(min(args.n_train, len(train_pool)), random_state=seed)
        _, mav = fit_maveriq(tr)
        preds = {'EnergyLens': unified_predict(fit_unified(tr), test),
                 'MaverIQ': maveriq_energy(mav, test),
                 'Vidur': test['vidur_latency'].to_numpy() * test['gpu_power_w'].to_numpy()
                          / np.maximum(test['batch_size'].to_numpy(), 1)}
        # without the oracle: each layout's median power over the 50 training rows
        p_med = tr.groupby(['tp_degree', 'pp_degree'])['gpu_power_w'].median()
        pw = np.array([p_med.get((t, q), np.nan) for t, q in zip(test.tp_degree, test.pp_degree)])
        preds['Vidur, training power'] = (test['vidur_latency'].to_numpy() * pw
                                          / np.maximum(test['batch_size'].to_numpy(), 1))
        if _SKLEARN:
            preds['Gradient Boosting'] = fit_gb(tr)(test)
        y = test['measured_energy'].to_numpy()
        rk = ranking(test, {m: pd.Series(p, index=test.index) for m, p in preds.items()})
        for m, p in preds.items():
            rows.append({'seed': seed, 'method': m, 'mape': 100 * np.mean(np.abs(p - y) / y), **rk[m]})
    res = pd.DataFrame(rows).groupby('method')[['mape', 'pair', 'top1', 'regret']].mean().round(2)
    os.makedirs(args.out, exist_ok=True)
    name = os.path.basename(os.path.dirname(os.path.dirname(args.grid))) + '_' + os.path.basename(os.path.dirname(args.grid))
    res.to_csv(os.path.join(args.out, f'vidur_eval_{name}.csv'))
    print(res.to_string())


if __name__ == '__main__':
    main()
