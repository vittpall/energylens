#!/usr/bin/env python3
"""
Configuration selection vs simple rules, where the choice is and is not trivial.

Per held-out scenario (max_tokens, batch_size) each method picks a (TP, PP):
  unconstrained     any configuration
  fixed GPU count   only layouts with the same TP x PP (2 GPUs: (1,2)/(2,1);
                    4 GPUs: (1,4)/(2,2)/(4,1)) — "fewest GPUs" cannot decide here
  SLO               the lowest-energy configuration whose latency is at most
                    k x the scenario's fastest configuration (--slo k ...), or,
                    as in MaverIQ/Proteus, k x the latency of the configuration
                    with minimum parallelism (--slo-minpar k ..., default
                    2.5 5 7.5); "binds" = share of scenarios whose unconstrained
                    energy optimum violates the limit:
                    known feasibility      methods are told which configurations
                                           meet the limit and only rank energy
                    predicted feasibility  every method uses the same latency
                                           prediction (fit_latency: prefill +
                                           decode waves); a pick slower than the
                                           limit is a violation
Models (EnergyLens, MaverIQ, Gradient Boosting) are fit on the same n training
rows from the training scenarios (pairwise_ranking.py's fitters). Rules need no
fit; the "measured" ones read the held-out latency, so they are oracles. The
"fewest GPUs, then TP" rule is left out unless --with-rule is given.

    python test/selection_baselines.py --datasets "Label=path.csv" ... --n-train 50 0
"""
import argparse, os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_args import add_datasets_arg, resolve_datasets
from compare_cross_model import holdout_scenarios
from pairwise_ranking import (DATASETS, prepare_df, fit_unified, unified_predict,
                              fit_maveriq, maveriq_energy, fit_gb, fit_latency, _SKLEARN)

SCEN = ['max_tokens', 'batch_size']
CFG = ['tp_degree', 'pp_degree']
KNOWN, PREDICTED = 'SLO, known feasibility', 'SLO, predicted feasibility'
MINPAR = ' (min-parallelism base)'


def fewest_gpus(x):
    return x.sort_values(['gpus', 'pp_degree']).index[0]


def rules(x, with_rule=False):
    """Picks (row labels of x) of rules that need no fitted model."""
    return {
        **({'fewest GPUs, then TP': fewest_gpus(x)} if with_rule else {}),
        'max parallelism': x.sort_values(['gpus', 'pp_degree'], ascending=[False, True]).index[0],
        'prefer TP': x.sort_values(['tp_degree', 'pp_degree'], ascending=[False, True]).index[0],
        'prefer PP (Accelerate: layers over all GPUs)':
            x.sort_values(['pp_degree', 'tp_degree'], ascending=[False, True]).index[0],
        'lowest latency (measured)': x.lat.idxmin(),
        'latency x GPUs (measured)': (x.lat * x.gpus).idxmin(),
        # the same two choices with the latency predicted from the training rows, as the
        # serving systems would make them
        'lowest predicted latency (Vidur-style search)': x.plat.idxmin(),
        'predicted latency x GPUs (AlpaServe-style throughput)': (x.plat * x.gpus).idxmin(),
    }


def per_config(test, preds, pred_lat):
    t = test.assign(lat=test.measured_latency, plat=pred_lat,
                    **{f'p_{m}': p for m, p in preds.items()})
    agg = t.groupby(SCEN + CFG).mean(numeric_only=True).reset_index()
    agg['gpus'] = agg.tp_degree * agg.pp_degree
    return agg


def evaluate(agg, methods, setting, with_rule=False):
    """Top-1 / regret over the choice sets of the unconstrained or fixed-GPU setting."""
    keys = SCEN + (['gpus'] if setting == 'fixed GPU count' else [])
    out = []
    for _, x in agg.groupby(keys):
        if len(x) < 2:
            continue
        y, best = x.measured_energy, x.measured_energy.min()
        picks = {m: (x[f'p_{m}'].idxmin() if x[f'p_{m}'].notna().any() else x.lat.idxmin()) for m in methods}
        picks.update(rules(x, with_rule))
        for m, i in picks.items():
            out.append({'setting': setting, 'k': '-', 'method': m, 'top1': float(y[i] == best),
                        'regret': (y[i] - best) / best * 100, 'violation': 0.0})
    return out


def evaluate_slo(agg, methods, ks, keys=SCEN, with_rule=False, base='fastest'):
    """Lowest energy under a latency limit, with known or predicted feasibility.
    agg: one row per (scenario, config) with lat (measured), plat (predicted),
    gpus, measured_energy and p_<method> energy predictions. base: 'fastest'
    (limit = k x fastest configuration) or 'minpar' (k x the configuration with
    the fewest GPUs, then TP, as MaverIQ/Proteus define latency SLOs)."""
    out = []
    for _, x in agg.groupby(keys):
        if len(x) < 2:
            continue
        ref = x.lat.min() if base == 'fastest' else x.lat[fewest_gpus(x)]
        for k in ks:
            slo = k * ref
            feasible = x.lat <= slo
            best = x.measured_energy[feasible].min()
            binds = float(x.lat[x.measured_energy.idxmin()] > slo)
            for mode in (KNOWN, PREDICTED):
                lat = x.lat if mode == KNOWN else x.plat
                ok = lat <= slo
                c = x[ok] if ok.any() else x.loc[[lat.idxmin()]]
                picks = {m: (c[f'p_{m}'].idxmin() if c[f'p_{m}'].notna().any() else lat.idxmin()) for m in methods}
                if with_rule:
                    picks['fewest GPUs, then TP'] = fewest_gpus(c)
                picks['fastest'] = lat.idxmin()
                picks['predicted latency x GPUs (AlpaServe-style throughput)'] = (c.plat * c.gpus).idxmin()
                if mode == KNOWN:
                    picks['latency x GPUs (measured)'] = (c.lat * c.gpus).idxmin()
                for m, i in picks.items():
                    viol = bool(x.lat[i] > slo)
                    e = x.measured_energy[i]
                    out.append({'setting': mode + ('' if base == 'fastest' else MINPAR), 'k': k,
                                'method': m, 'top1': float(not viol and e == best),
                                'regret': np.nan if viol else (e - best) / best * 100,
                                'violation': float(viol), 'binds': binds})
    return out


def lookup_predict(train, test):
    """DynamoLLM / ML.ENERGY-style lookup, no model: each test row takes the energy of the
    nearest profiled row of the same (TP, PP, budget) configuration, in log batch size,
    output length and input per request. Configurations with no profiled row cannot be
    predicted (NaN), so they are never chosen."""
    def feats(d):
        return np.column_stack([np.log(d.batch_size), np.log(np.maximum(d.max_tokens, 1)),
                                np.log1p(d.total_input_tokens / np.maximum(d.batch_size, 1))])
    out = np.full(len(test), np.nan)
    bud = 'max_num_batched_tokens' if 'max_num_batched_tokens' in test else None
    keys = CFG
    ftr, fte = feats(train), feats(test)
    scale = ftr.std(axis=0) + 1e-9
    tr_groups = train.groupby(keys).indices
    for key, idx in test.groupby(keys).indices.items():
        j = tr_groups.get(key)
        if j is None:
            continue
        d = (((fte[idx][:, None, :] - ftr[j][None, :, :]) / scale) ** 2).sum(-1)
        out[idx] = train.measured_energy.to_numpy()[j][d.argmin(1)]
    return out


def run(name, path, n_train, seed, ks, with_rule=False, ks_minpar=()):
    df = prepare_df(pd.read_csv(path), path)
    train, test, n_test, _ = holdout_scenarios(df, 0.3, seed)
    if n_test == 0:
        return []
    if 0 < n_train < len(train):
        train = train.iloc[np.random.RandomState(seed + 1).choice(len(train), n_train, replace=False)]
    _, lat_fn = fit_maveriq(train)
    preds = {'EnergyLens': unified_predict(fit_unified(train), test),
             'MaverIQ': maveriq_energy(lat_fn, test)}
    if _SKLEARN:
        preds['Gradient Boosting'] = fit_gb(train)(test)
    preds['lookup, nearest profiled row (DynamoLLM-style)'] = lookup_predict(train, test)
    plat = fit_latency(train)(test)
    y = test['measured_latency'].values
    print(f'    latency MAPE on held-out rows: {100 * np.mean(np.abs(plat - y) / y):.1f}%', flush=True)
    agg = per_config(test, preds, plat)
    rows = (evaluate(agg, preds, 'unconstrained', with_rule)
            + evaluate(agg, preds, 'fixed GPU count', with_rule))
    rows += evaluate_slo(agg, preds, ks, with_rule=with_rule)
    rows += evaluate_slo(agg, preds, ks_minpar, with_rule=with_rule, base='minpar')
    return [{'dataset': name, 'n_train': n_train or 'all', 'seed': seed, **r} for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    add_datasets_arg(ap)
    ap.add_argument('--n-train', type=int, nargs='+', default=[50, 0], help='0 = all training rows')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 7, 123])
    ap.add_argument('--slo', type=float, nargs='+', default=[1.1, 1.25, 1.5, 2.0],
                    help='latency limits as multiples of the fastest configuration')
    ap.add_argument('--slo-minpar', type=float, nargs='*', default=[2.5, 5.0, 7.5],
                    help='latency limits as multiples of the minimum-parallelism configuration '
                         '(MaverIQ/Proteus definition)')
    ap.add_argument('--with-rule', action='store_true',
                    help="include the 'fewest GPUs, then TP' rule")
    ap.add_argument('--output-dir', default='results_figures')
    args = ap.parse_args()
    datasets = {k: v for k, v in resolve_datasets(args.datasets, DATASETS).items() if os.path.isfile(v)}

    rows = []
    for name, path in datasets.items():
        for n in args.n_train:
            for seed in args.seeds:
                print(f'  {name}  n_train={n or "all"}  seed={seed}', flush=True)
                rows += run(name, path, n, seed, args.slo, args.with_rule, args.slo_minpar)
    r = pd.DataFrame(rows)
    r['n_train'] = r['n_train'].astype(str)
    r['k'] = r['k'].astype(str)
    os.makedirs(args.output_dir, exist_ok=True)
    r.to_csv(os.path.join(args.output_dir, 'selection_baselines_detail.csv'), index=False)

    keys = ['setting', 'k', 'n_train', 'method']
    r['binds'] = r['binds'].fillna(0.0) if 'binds' in r else 0.0
    per_seed = r.groupby(keys + ['seed'])[['top1', 'regret', 'violation', 'binds']].mean()
    summary = per_seed.groupby(keys).agg(['mean', 'std'])
    for c in ('top1', 'violation', 'binds'):
        summary[(c, 'mean')] *= 100
        summary[(c, 'std')] *= 100
    summary.round(2).to_csv(os.path.join(args.output_dir, 'selection_baselines_summary.csv'))
    for (setting, k, n), s in summary.groupby(level=[0, 1, 2]):
        s = s.droplevel([0, 1, 2]).sort_values(('regret', 'mean'))
        base = 'min-parallelism' if setting.endswith(MINPAR) else 'fastest'
        print(f'\n== {setting}' + (f', limit = {k} x {base}' if k != '-' else '') + f', n_train={n}'
              + (f'   (limit binds in {s[("binds", "mean")].iloc[0]:.1f}% of scenarios)' if k != '-' else ''))
        for m, v in s.iterrows():
            line = (f'   {m:<28} Top-1 {v[("top1", "mean")]:5.1f} ± {v[("top1", "std")]:4.1f}   '
                    f'regret {v[("regret", "mean")]:6.2f} ± {v[("regret", "std")]:5.2f}')
            if setting.startswith('SLO'):
                line += f'   violations {v[("violation", "mean")]:5.1f}%'
            print(line)


if __name__ == '__main__':
    main()
