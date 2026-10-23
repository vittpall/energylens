#!/usr/bin/env python3
"""Predict energy of real Azure request batches from the profiling grid.

    python test/trace_eval.py \
        --grid ./results_zeus/run_text_only_Mistral-7B-Instruct-v0.3_fp16/res_native/profiling_metrics.csv \
        --trace ./results_zeus_trace/run_trace_Mistral-7B-Instruct-v0.3_fp16/profiling_metrics.csv
"""
import argparse, os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pairwise_ranking import (prepare_df, fit_unified, unified_predict, fit_maveriq,
                              maveriq_energy, fit_gb, fit_latency, compute_pairwise_metrics,
                              _SKLEARN)
from online_eval import evaluate_slo
from compare_cross_model import drop_invalid_rows

def load_trace(path):
    raw = pd.read_csv(path)
    raw = raw[pd.to_numeric(raw['energy'], errors='coerce') > 0].reset_index(drop=True)
    # drop invalid rows here, so the raw columns copied below stay aligned with df
    raw = drop_invalid_rows(raw, path)
    df = prepare_df(raw, path)
    df['batch_id'], df['trace'] = raw['batch_id'].values, raw['trace'].values
    return df


def predict(train, test):
    _, lat_fn = fit_maveriq(train)
    preds = {'EnergyLens': unified_predict(fit_unified(train), test),
             'MaverIQ': maveriq_energy(lat_fn, test)}
    if _SKLEARN:
        preds['Gradient Boosting'] = fit_gb(train)(test)
    return preds


def score_slo(test, preds, plat, ks):
    agg = test.assign(lat=test['measured_latency'], plat=plat,
                      gpus=test['tp_degree'] * test['pp_degree'],
                      **{f'p_{m}': p for m, p in preds.items()})
    y = test['measured_latency'].values
    lat_mape = 100 * np.mean(np.abs(plat - y) / y)
    return ([{'setting': 'latency', 'k': '-', 'method': 'fit_latency', 'metric': 'mape',
              'value': lat_mape}]
            + [{'setting': r['setting'], 'k': r['k'], 'method': r['method'], 'metric': c,
                'value': r[c] * (100 if c != 'regret' else 1)}
               for r in evaluate_slo(agg, preds, ks, keys=['batch_id'])
               for c in ('top1', 'regret', 'violation')])


def score(test, preds):
    y = test['measured_energy'].values
    out = [{'method': m, 'metric': 'mape', 'value': 100 * np.mean(np.abs(p - y) / y)}
           for m, p in preds.items()]
    for trace in test['trace'].unique():
        sel = (test['trace'] == trace).values
        out += [{'method': m, 'metric': f'mape_{trace}',
                 'value': 100 * np.mean(np.abs(p[sel] - y[sel]) / y[sel])} for m, p in preds.items()]
    rank = []
    gpus = (test['tp_degree'] * test['pp_degree']).values
    for _, idx in test.groupby('batch_id').indices.items():
        labels = list(zip(test['tp_degree'].values[idx], test['pp_degree'].values[idx]))
        p = {m: v[idx] for m, v in preds.items()}
        for m, r in compute_pairwise_metrics(y[idx], p, labels).items():
            rank.append({'method': m, 'pairwise': 100 * r['pairwise_acc'],
                         'top1': 100 * r['top1_correct'], 'regret': r['regret_pct']})
    rank = pd.DataFrame(rank).groupby('method').mean()
    out += [{'method': m, 'metric': c, 'value': rank.loc[m, c]}
            for m in rank.index for c in rank.columns]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--grid', required=True, help='grid profiling_metrics.csv of the same model')
    ap.add_argument('--trace', required=True, help='trace-mode profiling_metrics.csv')
    ap.add_argument('--n-train', type=int, nargs='+', default=[50, 0], help='grid rows; 0 = all')
    ap.add_argument('--adapt', type=int, nargs='+', default=[0, 10, 25],
                    help='calibration trace rows added to the grid rows')
    ap.add_argument('--balance', action=argparse.BooleanOptionalAction, default=True,
                    help='also fit grid + k trace with the trace rows up-weighted to the grid size')
    ap.add_argument('--slo', type=float, nargs='+', default=[1.1, 1.25, 1.5, 2.0],
                    help='latency limits as multiples of the batch\'s fastest layout')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 7, 123])
    ap.add_argument('--output-dir', default='results_figures')
    args = ap.parse_args()

    grid, trace = prepare_df(pd.read_csv(args.grid), args.grid), load_trace(args.trace)
    print(f'grid: {len(grid)} rows, input/request median {np.median(grid.total_input_tokens / grid.batch_size):.0f}')
    print(f'trace: {len(trace)} rows, {trace.batch_id.nunique()} batches, input/request median '
          f'{np.median(trace.total_input_tokens / trace.batch_size):.0f}')

    rows, slo_rows = [], []
    for seed in args.seeds:
        rng = np.random.RandomState(seed)
        ids = rng.permutation(trace['batch_id'].unique())
        calib = trace[trace['batch_id'].isin(ids[:len(ids) // 2])]
        test = trace[trace['batch_id'].isin(ids[len(ids) // 2:])].reset_index(drop=True)
        runs = {'trace only': calib}
        for n in args.n_train:
            g = grid if n == 0 or n >= len(grid) else grid.iloc[rng.choice(len(grid), n, replace=False)]
            for k in args.adapt:
                extra = calib.iloc[rng.choice(len(calib), min(k, len(calib)), replace=False)]
                runs[f'grid n={n or "all"} + {k} trace'] = pd.concat([g, extra], ignore_index=True)
                if args.balance and k:
                    # Repeating rows = integer sample weights; none of the fits take weights.
                    rep = max(1, round(len(g) / len(extra)))
                    runs[f'grid n={n or "all"} + {k} trace (balanced)'] = pd.concat(
                        [g] + [extra] * rep, ignore_index=True)
        for name, train in runs.items():
            print(f'  seed {seed}: {name} ({len(train)} rows)', flush=True)
            preds = predict(train, test)
            rows += [{'seed': seed, 'train': name, **r} for r in score(test, preds)]
            slo_rows += [{'seed': seed, 'train': name, **r}
                         for r in score_slo(test, preds, fit_latency(train)(test), args.slo)]

    r = pd.DataFrame(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    r.to_csv(os.path.join(args.output_dir, 'trace_eval_detail.csv'), index=False)
    s = r.groupby(['train', 'metric', 'method'])['value'].agg(['mean', 'std']).reset_index()
    s.to_csv(os.path.join(args.output_dir, 'trace_eval_summary.csv'), index=False)
    table = s.pivot_table(index=['train', 'method'], columns='metric', values='mean')
    cols = [c for c in ['mape'] + sorted(c for c in table.columns if c.startswith('mape_'))
            + ['pairwise', 'top1', 'regret'] if c in table.columns]
    print('\nHeld-out trace batches (mean over seeds; MAPE/pairwise/Top-1 in %, regret in %):')
    print(table[cols].round(1).to_string())


    # SLO: per-batch picks averaged per seed, then over seeds (NaN regret = violation)
    q = pd.DataFrame(slo_rows)
    q['k'] = q['k'].astype(str)
    q.to_csv(os.path.join(args.output_dir, 'trace_slo_detail.csv'), index=False)
    per_seed = q.groupby(['setting', 'k', 'train', 'method', 'metric', 'seed'])['value'].mean()
    qs = per_seed.groupby(level=[0, 1, 2, 3, 4]).agg(['mean', 'std']).reset_index()
    qs.to_csv(os.path.join(args.output_dir, 'trace_slo_summary.csv'), index=False)
    lat = qs[qs.setting == 'latency'].set_index('train')['mean']
    print('\nLatency predictor (fit_latency) MAPE on held-out trace batches, %:')
    print(lat.round(1).to_string())
    for setting in ('SLO, known feasibility', 'SLO, predicted feasibility'):
        t = qs[qs.setting == setting].pivot_table(index=['train', 'method'],
                                                   columns=['k', 'metric'], values='mean')
        cols = [(k, m) for k in sorted(t.columns.levels[0], key=float)
                for m in ('top1', 'regret', 'violation')
                if (k, m) in t.columns and not (setting.startswith('SLO, known') and m == 'violation')]
        print(f'\n{setting} (columns: limit k x fastest / Top-1 %, regret %, violations %):')
        print(t[cols].round(1).to_string())


if __name__ == '__main__':
    main()
