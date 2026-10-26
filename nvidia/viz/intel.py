#!/usr/bin/env python3
"""Figures for the Intel GPU (Intel Data Center GPU Max 1550) runs.

    python viz/intel.py [--out results_figures/intel]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NVIDIA = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(NVIDIA, "test"))
sys.path.insert(0, os.path.join(NVIDIA, "..", "adaptive_sampling"))
from style import plt, save  # noqa: E402
from datasets import C  # noqa: E402
from intel_eval import RUNS, load, TDP  # noqa: E402

COLOR = {"Mistral-7B": C["Mistral-7B"], "Llama-3.1-8B": C["Llama-3.1-8B"], "Qwen2.5-14B": C["Qwen2.5-14B"],
         "Qwen2-VL-7B (image)": C["Qwen2-VL-7B"], "Qwen1.5-MoE-A2.7B": C["Qwen1.5-MoE"]}
MARK = {"Mistral-7B": "o", "Llama-3.1-8B": "s", "Qwen2.5-14B": "D", "Qwen2-VL-7B (image)": "^", "Qwen1.5-MoE-A2.7B": "v"}
METHOD_COLOR = {"EnergyLens": "#1f77b4", "MaverIQ": "#ff7f0e", "Gradient Boosting": "#2ca02c"}


def per_layout(d):
    g = d.groupby(["tp_degree", "pp_degree"]).agg(lat=("measured_latency", "mean"), e=("measured_energy", "mean"),
                                                 p=("gpu_power_w", "mean")).reset_index()
    g["tiles"] = g.tp_degree * g.pp_degree
    return g.sort_values(["tiles", "pp_degree"]).reset_index(drop=True)


def characterization(out):
    data = {n: per_layout(load(a)) for n, (a, _) in RUNS.items()}
    a100 = {n: per_layout(load(z)) for n, (_, z) in RUNS.items()}
    layouts = sorted({(int(r.tp_degree), int(r.pp_degree)) for g in data.values() for r in g.itertuples()},
                     key=lambda l: (l[0] * l[1], l[1]))
    pos = {l: i for i, l in enumerate(layouts)}
    labels = [f"({t},{p})" for t, p in layouts]
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5))
    (a, b), (c, e) = axs
    for name, g in data.items():
        x = [pos[(int(t), int(p))] for t, p in zip(g.tp_degree, g.pp_degree)]
        kw = dict(color=COLOR[name], marker=MARK[name], ms=7, mec="white", lw=1.4, label=name)
        a.plot(x, g.lat, **kw)
        b.plot(x, g.e, **kw)
        c.plot(x, g.p, **kw)
    # card TDP: a layout of n tiles occupies ceil(n / 2) cards
    cards = [int(np.ceil(t * p / 2)) for t, p in layouts]
    c.step(range(len(layouts)), [TDP * n for n in cards], where="mid", color="grey", ls="--", lw=1.2)
    c.text(len(layouts) - 1, TDP * 6 + 60, "600 W TDP x cards used", color="grey", fontsize=11, ha="right")
    for ax, yl, t in ((a, "Batch latency (s)", "(a) Latency"), (b, "Energy per request (J)", "(b) Energy"),
                      (c, "Total power (W)", "(c) Power")):
        ax.set_xticks(range(len(layouts)), labels, rotation=45)
        ax.set_xlabel("(TP, PP), one rank per tile")
        ax.set_ylabel(yl)
        ax.set_title(t)
    a.set_yscale("log")
    b.set_yscale("log")
    a.legend(fontsize=10)
    # (d) Intel GPU / A100 energy per generated token, median over the grid cells both machines measured
    # (per token: the image prompts stop at EOS, and the Intel node's answers are longer than the A100's)
    k = ["tensor_parallel_size", "pipeline_parallel_size", "max_tokens", "batch_size", "max_num_batched_tokens",
         "input_length"]

    def ept(path):
        d = pd.read_csv(os.path.join(NVIDIA, path))
        return (d.energy * d.batch_size / d.generated_tokens).groupby([d[c] for c in k]).mean()
    ratio = {}
    for name, (ap, zp) in RUNS.items():
        j = pd.concat([ept(ap).rename("a"), ept(zp).rename("z")], axis=1, join="inner")
        ratio[name] = (j.a / j.z).groupby(level=[0, 1]).median()
    shared = sorted({l for r in ratio.values() for l in r.index}, key=lambda l: (l[0] * l[1], l[1]))
    w = 0.8 / len(ratio)
    rows = []
    for i, (name, r) in enumerate(ratio.items()):
        xs = [j for j, l in enumerate(shared) if l in r.index]
        vals = [r[shared[j]] for j in xs]
        e.bar(np.array(xs) + (i - (len(ratio) - 1) / 2) * w, vals, w, color=COLOR[name], label=name)
        rows += [{"dataset": name, "tp": shared[j][0], "pp": shared[j][1], "energy_ratio": v} for j, v in zip(xs, vals)]
    e.axhline(1, color="black", lw=1)
    e.set_xticks(range(len(shared)), [f"({t},{p})" for t, p in shared])
    e.set_xlabel("(TP, PP)")
    e.set_ylabel("Energy per output token, Intel GPU / A100")
    e.set_title("(d) Intel GPU / A100, same layout and workload")
    fig.tight_layout()
    fig._laid_out = True
    save(fig, out, "intel_characterization")
    res = pd.concat([pd.concat([g.assign(dataset=n, gpu="Max 1550") for n, g in data.items()]),
                     pd.concat([g.assign(dataset=n, gpu="A100") for n, g in a100.items()])])
    res.to_csv(os.path.join(out, "intel_per_layout.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(out, "intel_vs_a100.csv"), index=False)
    return res


def side_by_side(out):
    """Same model on both machines: latency, energy per output token and power per layout, each
    averaged over the workload cells the A100 grid measured (the Intel GPU grid contains it)."""
    k = ["max_tokens", "batch_size", "max_num_batched_tokens", "input_length"]
    fig, axs = plt.subplots(len(RUNS), 3, figsize=(16, 3.6 * len(RUNS)), squeeze=False)
    rows = []
    for i, (name, (ap, zp)) in enumerate(RUNS.items()):
        raw = {g: pd.read_csv(os.path.join(NVIDIA, p)) for g, p in (("Max 1550", ap), ("A100", zp))}
        cells = set(map(tuple, raw["A100"][k].drop_duplicates().to_numpy()))
        per = {}
        for g, d in raw.items():
            d = d[d[k].apply(tuple, axis=1).isin(cells)]
            d = d.assign(ept=d.energy * d.batch_size / d.generated_tokens)
            m = d.groupby(["tensor_parallel_size", "pipeline_parallel_size"]).agg(
                lat=("latency", "mean"), ept=("ept", "mean"), p=("power", "mean")).reset_index()
            m["n"] = m.tensor_parallel_size * m.pipeline_parallel_size
            per[g] = m
            rows += [{"dataset": name, "gpu": g, **r} for r in m.to_dict("records")]
        lays = sorted({(int(r.tensor_parallel_size), int(r.pipeline_parallel_size)) for m in per.values()
                       for r in m.itertuples()}, key=lambda l: (l[0] * l[1], l[1]))
        pos = {l: j for j, l in enumerate(lays)}
        for g, (col, mk) in {"A100": ("#76b900", "o"), "Max 1550": ("#0071c5", "s")}.items():
            m = per[g]
            x = [pos[(int(t), int(p))] for t, p in zip(m.tensor_parallel_size, m.pipeline_parallel_size)]
            o = np.argsort(x)
            for ax, v in zip(axs[i], ("lat", "ept", "p")):
                ax.plot(np.array(x)[o], m[v].to_numpy()[o], color=col, marker=mk, ms=7, mec="white", lw=1.6,
                        label=f"NVIDIA A100" if g == "A100" else "Intel Max 1550 (one rank per tile)")
        for ax, yl in zip(axs[i], ("Batch latency (s)", "Energy per output token (J)", "Total power (W)")):
            ax.set_xticks(range(len(lays)), [f"({t},{p})" for t, p in lays], rotation=45 if len(lays) > 8 else 0)
            ax.set_ylabel(yl)
            ax.set_title(name, fontsize=12)
        axs[i][0].set_yscale("log")
        axs[i][1].set_yscale("log")
    for ax in axs[-1]:
        ax.set_xlabel("(TP, PP)")
    axs[0][0].legend(fontsize=10)
    fig.tight_layout()
    fig._laid_out = True
    save(fig, out, "intel_vs_a100_models")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "intel_vs_a100_models.csv"), index=False)
    return res


GPU_STYLE = {"A100": dict(color="#76b900", marker="o", label="NVIDIA A100 (40 GB)"),
             "Max 1550": dict(color="#0071c5", marker="s", label="Intel Max 1550 (one rank per tile)")}
PAIR_MODELS = ("Llama-3.1-8B", "Qwen2-VL-7B (image)")


def pair_figure(out, models=PAIR_MODELS):
    """The condensed comparison on one row: latency, energy per output token and power (with each
    machine's maximum power draw) per layout; colour = GPU, line style = model, both GPUs on the same
    workload cells."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    d = pd.read_csv(os.path.join(out, "intel_vs_a100_models.csv"))
    d = d[d.dataset.isin(models)]
    lays = sorted({(int(t), int(p)) for t, p in zip(d.tensor_parallel_size, d.pipeline_parallel_size)},
                  key=lambda l: (l[0] * l[1], l[1]))
    pos = {l: j for j, l in enumerate(lays)}
    d = d.assign(x=[pos[(int(t), int(p))] for t, p in zip(d.tensor_parallel_size, d.pipeline_parallel_size)])
    model_ls = dict(zip(models, ("-", "--", "-.", ":")))
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    for name in models:
        for g, st in GPU_STYLE.items():
            s = d[(d.dataset == name) & (d.gpu == g)].sort_values("x")
            for ax, v in zip(axs, ("lat", "ept", "p")):
                ax.plot(s.x, s[v], color=st["color"], marker=st["marker"], ls=model_ls[name], ms=7, mec="white",
                        mew=1.0, lw=1.9)
    # maximum power draw: depends only on the devices a layout occupies (400 W per A100, 600 W per
    # Max 1550 card of two tiles), so one dotted line per device count over the layouts that use it
    c = axs[2]
    for g, st in GPU_STYLE.items():
        ls_ = d[d.gpu == g].drop_duplicates("x")
        n = ls_.tensor_parallel_size * ls_.pipeline_parallel_size
        dev = (n if g == "A100" else np.ceil(n / 2)).astype(int)
        for k, grp in ls_.assign(dev=dev.values).groupby("dev"):
            cap = (400.0 if g == "A100" else TDP) * k
            x0, x1 = grp.x.min() - 0.4, grp.x.max() + 0.4
            c.hlines(cap, x0, x1, color=st["color"], ls=":", lw=2)
            below = g == "A100"
            c.annotate(f"{k}$\\times$A100" if below else f"{k} card{'s' if k > 1 else ''}",
                       (x0 if below else x1, cap), xytext=(3 if below else -3, -3 if below else 3),
                       textcoords="offset points", color=st["color"], fontsize=10,
                       ha="left" if below else "right", va="top" if below else "bottom",
                       bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.8))
    for ax, yl, t in zip(axs, ("Batch latency (s)", "Energy per output token (J)", "Total power (W)"),
                         ("(a) Latency", "(b) Energy per output token", "(c) Power and maximum power draw")):
        ax.set_xticks(range(len(lays)), [f"{a},{b}" for a, b in lays], fontsize=11)
        ax.set_xlabel("Layout (TP, PP)")
        ax.set_ylabel(yl)
        ax.set_title(t)
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    c.set_ylim(200, None)
    c.yaxis.set_major_locator(FixedLocator([250, 500, 1000, 2000, 4000]))
    c.yaxis.set_major_formatter(ScalarFormatter())
    c.yaxis.set_minor_locator(NullLocator())
    h = [Line2D([], [], color=st["color"], marker=st["marker"], ls="-", lw=2, label=st["label"])
         for st in GPU_STYLE.values()]
    h += [Line2D([], [], color="black", ls=model_ls[m], lw=1.6, label=m) for m in models]
    h += [Line2D([], [], color="grey", ls=":", lw=2, label="max power draw (400 W per A100, 600 W per card)")]
    fig.legend(handles=h, loc="upper center", ncol=5, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig._laid_out = True
    save(fig, out, "intel_vs_a100")


def _binned_power(d, tp, pp, bins):
    g = d[(d.tensor_parallel_size == tp) & (d.pipeline_parallel_size == pp)]
    b = pd.cut(g.batch_size, bins)
    m = g.groupby(b, observed=True).agg(x=("batch_size", "median"), y=("power", "mean"), n=("power", "size"))
    return m[m.n >= 3]


def power_pair(out, pareto="Llama-3.1-8B"):
    """power.png for both machines, on the models profiled on both: (a) one device (one A100 vs one
    Max 1550 card = two tiles) and (b) the (4,1) layout, power against batch size with each
    machine's maximum power draw; (c) time-energy trade-off and Pareto frontier of one model."""
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(18, 5))
    bins = np.unique(np.round(np.geomspace(1, 512, 12)))
    ls = {"A100": "-", "Max 1550": "--"}
    raw = {(n, g): pd.read_csv(os.path.join(NVIDIA, p)) for n, (ap, zp) in RUNS.items()
           for g, p in (("Max 1550", ap), ("A100", zp))}
    rows = []
    for ax, lay in ((a, {"A100": (1, 1), "Max 1550": (2, 1)}), (b, {"A100": (4, 1), "Max 1550": (4, 1)})):
        for (n, g), d in raw.items():
            k = _binned_power(d, *lay[g], bins)
            if len(k):
                ax.plot(k.x, k.y, color=COLOR[n], marker=MARK[n], ls=ls[g], ms=6, mec="white", lw=1.6)
                rows += [{"panel": ax is a and "a" or "b", "dataset": n, "gpu": g, "batch": x, "power_w": y}
                         for x, y in zip(k.x, k.y)]
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Batch size (requests)")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    caps = {a: {"A100": (400, "1 A100"), "Max 1550": (600, "1 Max 1550 card")},
            b: {"A100": (1600, "4 A100s"), "Max 1550": (1200, "2 Max 1550 cards")}}
    for ax, cc in caps.items():
        for g, (w, lab) in cc.items():
            col = GPU_STYLE[g]["color"]
            ax.axhline(w, color=col, ls=":", lw=1.6)
            ax.text(0.98, w, f"{lab}: TDP {w:,} W", color=col, fontsize=11, va="center", ha="right",
                    transform=ax.get_yaxis_transform(), bbox=dict(fc="white", ec="none", pad=1.5))
        ax.set_ylim(0, max(w for w, _ in cc.values()) * 1.18)
    a.set_ylabel("Power draw (W)")
    a.set_title("(a) One device: A100 (1,1) vs. Max 1550 card (2,1)")
    b.set_title("(b) Layout (4,1): 4 A100s vs. 4 tiles (2 cards)")
    # (c) every configuration of one model, as power.png (c), both machines
    k = ["max_tokens", "input_length"]
    for g in ("A100", "Max 1550"):
        d = raw[(pareto, g)]
        d = d[(d.max_tokens == 256) & ((d.input_length / d.batch_size).round(-1) == 70)]
        m = d.groupby(["tensor_parallel_size", "pipeline_parallel_size", "batch_size"], as_index=False)[
            ["latency", "energy", "generated_tokens"]].mean()
        m["tpot"] = m.latency / (m.generated_tokens / m.batch_size)     # per output token of one request
        st = GPU_STYLE[g]
        c.scatter(m.tpot, m.energy, color=st["color"], marker=st["marker"], s=34, edgecolor="white", lw=0.6,
                  alpha=0.75, label=st["label"])
        srt = m.sort_values(["tpot", "energy"])
        f = srt[srt.energy < srt.energy.cummin().shift(fill_value=np.inf)]
        c.plot(f.tpot, f.energy, color=st["color"], lw=2.2)
        best = m.loc[m.energy.idxmin()]
        c.annotate(f"min {best.energy:.0f} J\n({int(best.tensor_parallel_size)},{int(best.pipeline_parallel_size)}), "
                   f"batch {int(best.batch_size)}", (best.tpot, best.energy), xytext=(10, 22),
                   textcoords="offset points", fontsize=10, color=st["color"],
                   arrowprops=dict(arrowstyle="->", color=st["color"]))
        rows += [{"panel": "c", "dataset": pareto, "gpu": g, "tp": r.tensor_parallel_size,
                  "pp": r.pipeline_parallel_size, "batch": r.batch_size, "tpot_s": r.tpot, "energy_j": r.energy}
                 for r in m.itertuples()]
    c.set_xscale("log")
    c.set_yscale("log")
    c.set_xlabel("Time per output token (s)")
    c.set_ylabel("Energy per request (J)")
    c.set_title(f"(c) Time-energy trade-off, {pareto}")
    c.grid(alpha=0.25, which="both")
    c.spines[["top", "right"]].set_visible(False)
    c.legend(fontsize=10, loc="upper left")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLOR[n], marker=MARK[n], ls="-", label=n) for n in RUNS] + \
        [Line2D([], [], color="black", ls=ls[g], label=GPU_STYLE[g]["label"].split(" (")[0]) for g in ls]
    fig.legend(handles=h, loc="upper center", ncol=6, frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig._laid_out = True
    save(fig, out, "power_a100_vs_max1550")
    pd.DataFrame(rows).to_csv(os.path.join(out, "power_a100_vs_max1550.csv"), index=False)


def accuracy(out):
    acc = pd.read_csv(os.path.join(out, "accuracy.csv"))
    tr = pd.read_csv(os.path.join(out, "transfer.csv"))
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(17, 4.6))
    m = acc.groupby(["dataset", "method"]).mape.mean().unstack()
    names = list(RUNS)
    w = 0.8 / m.shape[1]
    for i, meth in enumerate(["EnergyLens", "MaverIQ", "Gradient Boosting"]):
        a.bar(np.arange(len(names)) + (i - 1) * w, m.loc[names, meth], w, color=METHOD_COLOR[meth], label=meth)
    a.set_xticks(range(len(names)), [n.replace(" (image)", "\n(image)") for n in names])
    a.set_ylabel("MAPE (%)")
    a.set_title("(a) 50-row fits on Intel GPUs, held-out scenarios")
    a.legend(fontsize=10)
    lo = pd.read_csv(os.path.join(out, "loco.csv")).pivot_table(index="dataset", columns="method", values="pairwise")
    for i, meth in enumerate(["EnergyLens", "MaverIQ", "Gradient Boosting"]):
        b.bar(np.arange(len(names)) + (i - 1) * w, lo.loc[names, meth], w, color=METHOD_COLOR[meth])
    b.axhline(50, color="grey", ls=":", lw=1)
    b.set_xticks(range(len(names)), [n.replace(" (image)", "\n(image)") for n in names])
    b.set_ylabel("Pairwise accuracy (%)")
    b.set_ylim(0, 100)
    b.set_title("(b) Unseen layout (leave one layout out)")
    t = tr.groupby(["method", "k"]).mape.mean().unstack(0)
    sty = {"A100 fit, rescaled": ("#9467bd", "s", "--"), "A100 fit as prior": ("#1f77b4", "o", "-"),
           "from scratch": ("#7f7f7f", "D", ":"), "Gradient Boosting": ("#2ca02c", "^", "-.")}
    for meth, (col, mk, ls) in sty.items():
        s = t[meth].dropna()
        c.plot(s.index, s.values, color=col, marker=mk, ls=ls, ms=8, mec="white", lw=1.6, label=meth)
    c.axhline(t["A100 fit, as is"].dropna().iloc[0], color="black", ls="--", lw=1, label="A100 fit, as is")
    c.set_yscale("log")
    c.set_xlabel("Intel GPU rows used (k)")
    c.set_ylabel("MAPE (%)")
    c.set_title("(c) Transfer of the A100 fit to Intel GPU")
    c.legend(fontsize=9.5)
    fig.tight_layout()
    fig._laid_out = True
    save(fig, out, "intel_accuracy")
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(NVIDIA, "results_figures", "intel"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(characterization(args.out).round(2).to_string())
    print(side_by_side(args.out).round(3).to_string())
    pair_figure(args.out)
    power_pair(args.out)
    if os.path.exists(os.path.join(args.out, "transfer.csv")):
        print(accuracy(args.out).round(1).to_string())


if __name__ == "__main__":
    main()
