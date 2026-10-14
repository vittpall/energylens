#!/usr/bin/env python3
"""Figure: live serving of the Azure traces (source/online_trace.py).

    python test/plot_online.py            # -> results_figures/online_serving.{png,pdf}
"""
import argparse, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_cross_model import LC_LABEL_SIZE, LC_TICK_SIZE, LC_LEGEND_SIZE

MODELS = [('Llama-3.1-8B', 'results_zeus_online/run_online_Llama-3.1-8B-Instruct_fp16'),
          ('Qwen2.5-14B', 'results_zeus_online/run_online_Qwen2.5-14B-Instruct_fp16')]
COLOR = {1: '#2a78d6', 2: '#eb6834', 4: '#1baf7a'}          # by GPU count
STYLE = {'single': ('-', 'o'), 'tp': ('-', 'o'), 'pp': ('--', 's'), 'both': ('-.', '^')}


def layout_style(tp, pp):
    kind = 'single' if tp * pp == 1 else 'tp' if pp == 1 else 'pp' if tp == 1 else 'both'
    ls, mk = STYLE[kind]
    return COLOR[tp * pp], ls, mk, f'TP{tp}×PP{pp}'


def load(path):
    d = pd.read_csv(os.path.join(path, 'profiling_metrics.csv'))
    d['J_req'] = d.energy / d.n_requests
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--trace', default='conv', help='trace shown in the rate panels')
    ap.add_argument('--output', default='results_figures/online_serving')
    args = ap.parse_args()
    data = {name: load(p) for name, p in MODELS}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={'width_ratios': [1, 1, 1.1]})
    layouts = sorted({(r.tensor_parallel_size, r.pipeline_parallel_size) for d in data.values()
                      for r in d.itertuples()}, key=lambda l: (l[0] * l[1], l[1]))
    for ax, (name, _) in zip(axes[:2], MODELS):
        d = data[name][data[name].trace == args.trace]
        g = d.groupby(['tensor_parallel_size', 'pipeline_parallel_size', 'rate']).J_req.mean()
        for tp, pp in layouts:
            if (tp, pp) not in g.index.droplevel(2):
                continue
            s = g.loc[(tp, pp)]
            c, ls, mk, lab = layout_style(tp, pp)
            ax.plot(s.index, s.values, ls=ls, marker=mk, color=c, lw=2, ms=8, mec='white', mew=1, label=lab)
        ax.set_xscale('log')
        ax.set_xticks([0.1, 0.3, 1.0])
        ax.set_xticklabels(['0.1×', '0.3×', '1×'])
        ax.minorticks_off()
        ax.set_xlabel('Arrival rate (× trace rate)', fontsize=LC_LABEL_SIZE - 1)
        ax.set_title(f'{name}, {"conversation" if args.trace == "conv" else "code"} trace',
                     fontsize=LC_LABEL_SIZE - 1)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel('Energy per request (J)', fontsize=LC_LABEL_SIZE - 1)

    # SLO trade-off at the trace's own rate (Qwen2.5-14B; Llama-3.1-8B is the same shape)
    ax = axes[2]
    name = MODELS[1][0]
    d = data[name][(data[name].trace == args.trace) & (data[name].rate == 1.0)]
    g = d.groupby(['tensor_parallel_size', 'pipeline_parallel_size'])[['J_req', 'p90_e2e_s']].mean()
    for (tp, pp), r in g.iterrows():
        c, _, mk, lab = layout_style(tp, pp)
        ax.plot(r.p90_e2e_s, r.J_req, marker=mk, color=c, ms=11, mec='white', mew=1, ls='none')
        below = tp * pp == 1                     # keeps it clear of TP1xPP2 just above-left
        ax.annotate(lab, (r.p90_e2e_s, r.J_req), textcoords='offset points', xytext=(0, -16 if below else 9),
                    ha='center', fontsize=LC_TICK_SIZE - 4, color='#333333')
    ax.set_xscale('log')
    ticks = [5, 10, 20, 50, 100]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(5, 110)
    ax.set_xlabel('p90 request latency (s, log scale)', fontsize=LC_LABEL_SIZE - 1)
    ax.set_ylabel('Energy per request (J)', fontsize=LC_LABEL_SIZE - 1)
    ax.set_title(f'{name} at 1×: energy vs latency', fontsize=LC_LABEL_SIZE - 1)
    ax.set_ylim(0, g.J_req.max() * 1.2)

    for ax in axes:
        ax.tick_params(labelsize=LC_TICK_SIZE - 1)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_facecolor('#f9f9f9')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles, labels, fontsize=LC_LEGEND_SIZE, ncol=6, loc='lower center', frameon=False)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{args.output}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {args.output}.png / .pdf')


if __name__ == '__main__':
    main()
