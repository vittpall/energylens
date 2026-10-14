#!/usr/bin/env python3
"""
Check our MaverIQ reimplementation against the authors' code on their released data.

Data: MaverIQ/outputs/profiling_data/full_model_data_collection.csv (RTX A6000,
TensorRT-LLM, batch size 1, balanced pipeline mapping, output lengths 10..200).
Per (model, precision) every (TP, PP, output length) row is predicted by:

  authors (full-model)   their closed form: TTFT/TPOT from output lengths 10 and
                         20 of the reference layouts, TP/PP overheads solved from
                         L(1,1), L(1,2), L(2,1) (or their per-model fallbacks), and
                         their published exponents (latency_exponents.csv)
  ours, reference rows   our latency model (pairwise_ranking.fit_maveriq) fitted on
                         the same reference layouts at output lengths 10 and 20
  ours, half the rows    our latency model fitted on a random half of the rows,
                         scored on the other half (five seeds)

Reports the median and mean absolute percentage error of latency (the paper's
Fig. 9 reports the median).

    python test/maveriq_reproduction.py --maveriq ~/MaverIQ
"""
import argparse, os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pairwise_ranking import fit_maveriq

LAYERS = {'falcon-7b': 32, 'falcon-40b': 60, 'gptj-6b': 28,
          'llama-2-7b': 32, 'llama-2-13b': 40, 'llama-2-70b': 80}


def balanced_map(layers, pp):
    base, rem = divmod(layers, pp)
    return ' '.join(str(base + (i < rem)) for i in range(pp))


def load(root):
    df = pd.read_csv(os.path.join(root, 'outputs/profiling_data/full_model_data_collection.csv'))
    df = df[df.Batch_Size == 1]
    df = df[df.apply(lambda r: r.PP_Map == balanced_map(LAYERS[r.Model], r.PP), axis=1)]
    return df.rename(columns={'Output_Lenght': 'out'})


# ── authors' full-model estimator (tools/estimator/GD_algorithm.py) ──
def _lat(g, tp, pp, out):
    sel = g[(g.TP == tp) & (g.PP == pp)].set_index('out').Latency
    tpot = (sel[20] - sel[10]) / 10
    return sel[10] - 10 * tpot + out * tpot


def authors_predict(g, model, dtype, exps, rows):
    a, b, c, d = exps
    preds = []
    for tp, pp, out in zip(rows.TP, rows.PP, rows.out):
        if model == 'falcon-7b':
            l1, l2 = _lat(g, 1, 1, out), _lat(g, 1, 2, out)
            ref, ppo, tpo = l1, l2 - l1 / 2 ** a, 0.0
        elif (model == 'falcon-40b' and dtype == 'float16') or (model == 'llama-2-70b' and dtype == 'int8'):
            l1, l2, l3 = _lat(g, 1, 2, out), _lat(g, 1, 3, out), _lat(g, 2, 1, out)
            ref = (l2 - 2 ** d * l1) / ((1 / 3) ** a - 2 ** (d - 1))
            ppo, tpo = l1 - ref / 2 ** a, l3 - ref / 2
        elif model == 'llama-2-70b' and dtype == 'float16':
            l1, l2, l3 = _lat(g, 1, 3, out), _lat(g, 1, 4, out), _lat(g, 2, 2, out)
            ref = (l2 - (3 / 2) ** d * l1) / ((1 / 4) ** a - 3 ** (d - a) / 2 ** d)
            ppo = (l1 - ref / 3 ** a) / 2 ** d
            tpo = 2 ** c * (l3 - ref / 2 ** (a + 1) - ppo)
        else:
            l1, l2, l3 = _lat(g, 1, 1, out), _lat(g, 1, 2, out), _lat(g, 2, 1, out)
            ref, ppo, tpo = l1, l2 - l1 / 2 ** a, l3 - l1 / 2
        ref, ppo, tpo = max(ref, 0), max(ppo, 0), max(tpo, 0)
        preds.append(ref / (tp * pp ** a) + (tp - 1) ** b / pp ** c * tpo + (pp - 1) ** d * ppo)
    return np.array(preds)


REFERENCE = {('falcon-7b', None): [(1, 1), (1, 2)],
             ('falcon-40b', 'float16'): [(1, 2), (1, 3), (2, 1)],
             ('llama-2-70b', 'int8'): [(1, 2), (1, 3), (2, 1)],
             ('llama-2-70b', 'float16'): [(1, 3), (1, 4), (2, 2)]}


def reference_layouts(model, dtype):
    return REFERENCE.get((model, dtype)) or REFERENCE.get((model, None)) or [(1, 1), (1, 2), (2, 1)]


# ── our reimplementation ──
def ours_frame(g):
    return pd.DataFrame({'tp_degree': g.TP.values, 'pp_degree': g.PP.values, 'batch_size': 1,
                         'decode_tokens': g.out.values.astype(float),
                         'measured_latency': g.Latency.values})


def ape(pred, y):
    return 100 * np.abs(pred - y) / y


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--maveriq', default=os.path.expanduser('~/MaverIQ'))
    ap.add_argument('--dtypes', nargs='+', default=['float16', 'int8', 'int4'])
    ap.add_argument('--output', default='results_figures/maveriq_reproduction.csv')
    args = ap.parse_args()

    df = load(args.maveriq)
    ex = pd.read_csv(os.path.join(args.maveriq, 'outputs/estimator_data/latency_exponents.csv'))
    ex = ex[ex.Method == 'full-model'].set_index('Quant-config')

    rows = []
    for (model, dtype), g in df[(df.INT8_KV_Cache == False) & df.DTYPE.isin(args.dtypes)].groupby(['Model', 'DTYPE']):
        y = g.Latency.values
        try:
            pa = ape(authors_predict(g, model, dtype, ex.loc[f'({dtype}, False)', ['alpha', 'beta', 'gamma', 'delta']].values, g), y)
        except KeyError:
            continue                                   # a reference layout was not profiled
        ref = reference_layouts(model, dtype)
        train = g[[(t, p) in ref for t, p in zip(g.TP, g.PP)] & g.out.isin([10, 20])]
        _, lat_ref = fit_maveriq(ours_frame(train))
        pr = ape(lat_ref(ours_frame(g)), y)
        ph = []
        for s in range(5):
            idx = np.random.RandomState(s).permutation(len(g))
            tr, te = g.iloc[idx[:len(g) // 2]], g.iloc[idx[len(g) // 2:]]
            _, lat_h = fit_maveriq(ours_frame(tr))
            ph.append(ape(lat_h(ours_frame(te)), te.Latency.values))
        ph = np.concatenate(ph)
        rows.append({'model': model, 'dtype': dtype, 'rows': len(g), 'layouts': g[['TP', 'PP']].drop_duplicates().shape[0],
                     'authors_median': np.median(pa), 'authors_mean': pa.mean(),
                     'ours_ref_median': np.median(pr), 'ours_ref_mean': pr.mean(),
                     'ours_half_median': np.median(ph), 'ours_half_mean': ph.mean()})
        print(f'  {model:<12} {dtype:<8} authors {np.median(pa):5.1f}%  ours(ref rows) {np.median(pr):5.1f}%  '
              f'ours(half) {np.median(ph):5.1f}%   (median APE)', flush=True)

    r = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    r.to_csv(args.output, index=False)
    print('\nMedian / mean APE over all (model, precision) pairs:')
    print(r[[c for c in r.columns if c.endswith(('_median', '_mean'))]].median().round(1).to_string())


if __name__ == '__main__':
    main()
