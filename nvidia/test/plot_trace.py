#!/usr/bin/env python3
"""Figure: held-out MAPE on real Azure request batches, per model and method.

    python test/plot_trace.py            # -> results_figures/trace_mape.{png,pdf}
"""
import argparse, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_cross_model import LC_LABEL_SIZE, LC_TICK_SIZE, LC_LEGEND_SIZE, LC_VALUE_SIZE, LC_COLORS

GROUPS = [   # (label, trace_eval output dir, grid it was fitted on)
    ('Mistral\n7B', 'results_figures/trace_Mistral-7B-Instruct-v0.3', 'std'),
    ('Mistral-Small\n24B', 'results_figures/trace_Mistral-Small-24B-Instruct-2501', 'std'),
    ('Qwen2.5\n14B', 'results_figures/trace_Qwen2.5-14B-Instruct', 'std'),
    ('Llama-3.1\n8B', 'results_figures/trace_Llama-3.1-8B-Instruct', 'std'),
    ('Qwen2.5\n32B', 'results_figures/trace_Qwen2.5-32B-Instruct', 'std'),
    ('Qwen2.5\n14B', 'results_figures/trace_Qwen2.5-14B-Instruct_inlen', 'inlen'),
    ('Llama-3.1\n8B', 'results_figures/trace_Llama-3.1-8B-Instruct_inlen', 'inlen'),
]
BLOCKS = {'std': 'standard grid (short prompts)',
          'inlen': 'grid with long prompts\n(128-4,096 tokens)'}
METHODS = [  # (trace_eval name, legend name as in the learning curves, hatch)
    ('EnergyLens', 'EnergyLens (Ours)', ''),
    ('MaverIQ', 'MaverIQ (Lat×P)', '//'),
    ('Gradient Boosting', 'Gradient Boosting', '..'),
]


def load(n, k):
    runs = {'grid': f'grid n={n} + 0 trace', 'adapt': f'grid n={n} + {k} trace (balanced)'}
    out = {}
    for label, d, _ in GROUPS:
        s = pd.read_csv(os.path.join(d, 'trace_eval_summary.csv'))
        s = s[s.metric == 'mape'].set_index(['train', 'method'])
        out[d] = {r: {m: s.loc[(t, m), ['mean', 'std']].values for m, _, _ in METHODS}
                      for r, t in runs.items()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--n-train', type=int, default=50)
    ap.add_argument('--adapt', type=int, default=25)
    ap.add_argument('--output', default='results_figures/trace_mape')
    args = ap.parse_args()
    data = load(args.n_train, args.adapt)

    titles = {'grid': f'{args.n_train} grid rows',
              'adapt': f'{args.n_train} grid rows + {args.adapt} real batches'}
    fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))
    x = np.arange(len(GROUPS))
    w = 0.26
    for ax, run in zip(axes, ('grid', 'adapt')):
        ymax = max(v[0] + v[1] for g in data.values() for v in g[run].values())
        for i, (m, name, hatch) in enumerate(METHODS):
            mean = np.array([data[d][run][m][0] for _, d, _ in GROUPS])
            std = np.array([data[d][run][m][1] for _, d, _ in GROUPS])
            bars = ax.bar(x + (i - 1) * w, mean, w * 0.92, yerr=std, capsize=3,
                          color=LC_COLORS[name], hatch=hatch, edgecolor='white', linewidth=0.8,
                          error_kw=dict(elinewidth=0.9, ecolor='#555555'), label=name, zorder=2)
            if m == 'EnergyLens':
                for b, v, e in zip(bars, mean, std):
                    ax.text(b.get_x() + b.get_width() / 2, v + e + ymax * 0.015, f'{v:.0f}',
                            ha='center', va='bottom', fontsize=LC_VALUE_SIZE, color='#222222')
        n_std = sum(k == 'std' for _, _, k in GROUPS)
        ax.axvline(n_std - 0.5, color='#999999', linestyle=':', linewidth=1)
        # which grid each block of bars was fitted on
        for k, (lo, hi) in {'std': (0, n_std - 1), 'inlen': (n_std, len(GROUPS) - 1)}.items():
            ax.text((lo + hi) / 2, 0.97, BLOCKS[k], transform=ax.get_xaxis_transform(), ha='center',
                    va='top', fontsize=LC_TICK_SIZE - 3, color='#444444', style='italic')
        ax.set_title(titles[run], fontsize=LC_LABEL_SIZE)
        ax.set_xticks(x)
        ax.set_xticklabels([g for g, _, _ in GROUPS], fontsize=LC_TICK_SIZE - 2)
        ax.tick_params(axis='y', labelsize=LC_TICK_SIZE)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_facecolor('#f9f9f9')
        ax.set_ylim(0, ymax * 1.22)    # room for the block headers
    axes[0].set_ylabel('MAPE on real batches (%)', fontsize=LC_LABEL_SIZE)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles, labels, fontsize=LC_LEGEND_SIZE, ncol=3, loc='lower center', frameon=False)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{args.output}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {args.output}.png / .pdf')


if __name__ == '__main__':
    main()
