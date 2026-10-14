#!/usr/bin/env python3
"""Characterization figures of the paper (Sec. V.B, "Empirical Foundations").

    python viz/characterization.py --out ../../PDP-paper/images/characterization   # from nvidia/
    python viz/characterization.py --only tp_pp batch                               # a subset

  tp_pp_vs_latency_avg / tp_pp_vs_energy_avg   every text-only model vs (TP, PP)
  energy_latency_optimum                        extra energy of the latency-optimal choice, per dataset
  batch_size_vs_energy                          (a) all models vs batch size, (b) one model per (TP, PP)
  modality_energy                               text vs image vs video, per (TP, PP)
  fits (CSV only)                               MaverIQ's latency form and EnergyLens' TP/PP terms
                                                fitted to the same data (numbers for Sec. V.B.2)

Each figure also writes a CSV of the plotted values next to it, which is where
the numbers quoted in the text come from. Colour = model, marker and line style
= architecture family (viz/style.py); datasets come from viz/datasets.py.
"""
import argparse
import os

import numpy as np
import pandas as pd

from datasets import select, get, NVIDIA
from style import (plt, style, save, line_kw, family_handles, FAMILY, FAMILY_COLOR, FAMILY_ORDER,
                   LEGEND_SIZE, LABEL_SIZE)

ORDER = [(1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (4, 1)]
XLABEL = "(Tensor Parallel, Pipeline Parallel)"
SCENARIO = ["max_tokens", "batch_size"]


def per_config(df, col):
    """Mean over the rows of the scenarios every configuration of this dataset measured,
    so each configuration is averaged over the same workloads (rows, not scenario means,
    are averaged, as for the numbers quoted in the paper)."""
    n = df[["tp_degree", "pp_degree"]].drop_duplicates().shape[0]
    common = df.groupby(SCENARIO).filter(lambda g: g[["tp_degree", "pp_degree"]].drop_duplicates().shape[0] == n)
    src = common if len(common) else df
    m = src.groupby(["tp_degree", "pp_degree"])[col].mean()
    return np.array([m.get(c, np.nan) for c in ORDER], float)


def _by_family(ds):
    return sorted(ds, key=lambda d: (FAMILY_ORDER.index(d.family), d.model))


def _tp_pp_axis(ax):
    x = np.arange(len(ORDER))
    ax.set_xticks(x)
    ax.set_xticklabels([f"({t}, {p})" for t, p in ORDER])
    return x


def tp_pp(out):
    """Latency and energy vs (TP, PP) for every text-only fp16 model."""
    ds = _by_family(select("text", models=None))
    ds = [d for d in ds if d.model not in ("Qwen2-VL-7B", "LLaVA-1.5-7B")]   # multimodal: see modality_energy
    rows = []
    for col, ylabel, name in (("measured_latency", "Average Batch Latency (s)", "tp_pp_vs_latency_avg"),
                              ("measured_energy", "Average Energy per Request (J)", "tp_pp_vs_energy_avg")):
        fig, ax = plt.subplots(figsize=(11, 7.5))
        x = _tp_pp_axis(ax)
        for d in ds:
            y = per_config(d.load(), col)
            ok = ~np.isnan(y)
            ax.plot(x[ok], y[ok], label=d.model, **line_kw(d.family, d.color))
            rows += [{"model": d.model, "family": d.family, "metric": col, "tp": t, "pp": p, "value": v}
                     for (t, p), v in zip(ORDER, y)]
        ax.set_yscale("log")
        style(ax, XLABEL, ylabel)
        ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.2), columnspacing=1.0, handlelength=2.6)
        save(fig, out, name)
    t = pd.DataFrame(rows)
    piv = t.pivot_table(index=["model", "family", "metric"], columns=["tp", "pp"], values="value")
    piv.to_csv(os.path.join(out, "tp_pp_avg.csv"))
    return piv


def optimum(out):
    """Extra energy of the lowest-latency configuration, per dataset (all fp16 and 4-bit runs)."""
    rows = []
    for d in select(precision=None):
        a = d.load().groupby(SCENARIO + ["tp_degree", "pp_degree"])[["measured_energy", "measured_latency"]] \
                    .mean().reset_index()
        n = diff = 0
        regret = []
        for _, g in a.groupby(SCENARIO):
            if len(g) < 2:
                continue
            n += 1
            e, l = g.measured_energy.idxmin(), g.measured_latency.idxmin()
            diff += e != l
            regret.append(100 * (g.measured_energy[l] / g.measured_energy[e] - 1))
        rows.append({"dataset": d.label, "family": d.family, "scenarios": n, "differ_pct": 100 * diff / n,
                     "regret_pct": np.mean(regret)})
    r = pd.DataFrame(rows).sort_values("regret_pct")
    r.to_csv(os.path.join(out, "energy_latency_optimum.csv"), index=False)

    fig, ax = plt.subplots(figsize=(11, 0.42 * len(r) + 1.5))
    y = np.arange(len(r))
    ax.barh(y, r.regret_pct, color=[FAMILY_COLOR[f] for f in r.family], height=0.65)
    for yi, (dv, rg) in enumerate(zip(r.differ_pct, r.regret_pct)):
        ax.text(rg + 1, yi, f"{rg:.0f}%  ({dv:.0f}%)", va="center", fontsize=LEGEND_SIZE - 1)
    ax.set_yticks(y)
    ax.set_yticklabels(r.dataset)
    style(ax, "Extra energy of the lowest-latency configuration (%)")
    ax.tick_params(axis="y", labelsize=LEGEND_SIZE)
    ax.set_xlim(0, r.regret_pct.max() * 1.35)
    from matplotlib.patches import Patch
    present = [f for f in FAMILY_ORDER if f in set(r.family)]
    ax.legend(handles=[Patch(color=FAMILY_COLOR[f], label=FAMILY[f][2]) for f in present],
              fontsize=LEGEND_SIZE, framealpha=0.9, loc="lower right")
    save(fig, out, "energy_latency_optimum")
    tot = r.scenarios.sum()
    print(f"optima differ in {(r.differ_pct * r.scenarios).sum() / 100:.0f}/{tot} scenarios "
          f"({(r.differ_pct * r.scenarios).sum() / tot:.1f}%); mean extra energy {r.regret_pct.mean():.1f}% "
          f"over {len(r)} datasets")
    return r


def _fewest_gpu_layout(df):
    lay = df[["tp_degree", "pp_degree"]].drop_duplicates()
    lay = lay.assign(g=lay.tp_degree * lay.pp_degree).sort_values(["g", "pp_degree"]).iloc[0]
    return int(lay.tp_degree), int(lay.pp_degree)


def _log_binned(df, col, bins):
    b = pd.cut(df.batch_size, bins)
    g = df.groupby(b, observed=True).agg(x=("batch_size", "median"), y=(col, "mean"), n=(col, "size"))
    return g[g.n >= 3]


def batch(out, detail_model="Qwen2-VL-7B"):
    """(a) every text-only model at its fewest-GPU layout, log-log; (b) one model per (TP, PP)."""
    ds = _by_family(select("text"))
    ds = [d for d in ds if d.model not in ("LLaVA-1.5-7B",)]
    bins = np.unique(np.round(np.geomspace(1, 512, 14)))
    fig, (a, b) = plt.subplots(1, 2, figsize=(17, 6.2))
    rows = []
    for d in ds:
        df = d.load()
        t, p = _fewest_gpu_layout(df)
        g = _log_binned(df[(df.tp_degree == t) & (df.pp_degree == p)], "measured_energy", bins)
        a.plot(g.x, g.y, label=f"{d.model} ({t},{p})", **line_kw(d.family, d.color))
        # sublinear batching exponent: slope of log E vs log b up to the concurrency limit (64)
        h = g[g.x <= 64]
        slope = np.polyfit(np.log(h.x), np.log(h.y), 1)[0] if len(h) >= 3 else np.nan
        rows.append({"model": d.model, "family": d.family, "layout": f"({t},{p})", "loglog_slope_b<=64": slope,
                     "E_smallest_bin": g.y.iloc[0], "E_largest_bin": g.y.iloc[-1],
                     "b_smallest_bin": g.x.iloc[0], "b_largest_bin": g.x.iloc[-1]})
    a.set_xscale("log")
    a.set_yscale("log")
    a.axvline(64, color="#555555", linestyle=":", linewidth=1)
    a.text(66, a.get_ylim()[0] * 1.3, "concurrency\nlimit S=64", fontsize=LEGEND_SIZE - 1, color="#555555")
    style(a, "Batch Size (requests)", "Average Energy per Request (J)")
    a.legend(fontsize=LEGEND_SIZE - 2, framealpha=0.9, ncol=2, loc="upper right")
    a.set_title("(a) All models, fewest-GPU configuration", fontsize=LABEL_SIZE)

    d = get(detail_model)
    df = d.load()
    colors = plt.cm.Set2(np.linspace(0, 1, 8))
    for (t, p), c in zip(ORDER, colors):
        g = _log_binned(df[(df.tp_degree == t) & (df.pp_degree == p)], "measured_energy", bins)
        b.plot(g.x, g.y, marker="o", markersize=7, linewidth=2, color=c, label=f"({t}, {p})",
               markeredgecolor="white", markeredgewidth=0.8)
    b.set_xscale("log")
    b.set_yscale("log")
    style(b, "Batch Size (requests)", "Average Energy per Request (J)")
    b.legend(fontsize=LEGEND_SIZE - 1, framealpha=0.9, title="(TP, PP)", ncol=2)
    b.set_title(f"(b) {detail_model}, every configuration", fontsize=LABEL_SIZE)
    save(fig, out, "batch_size_vs_energy")
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(out, "batch_size_vs_energy.csv"), index=False)
    return r


def modality(out):
    """Text vs image (vs video) energy per (TP, PP) for the multimodal models."""
    colors = {"text": "#4c72b0", "image": "#dd8452", "video": "#55a868", "video-only": "#8172b3"}
    models = ("Qwen2-VL-7B", "LLaVA-1.5-7B")
    fig, axes = plt.subplots(1, len(models), figsize=(15, 5.5))
    rows = []
    for ax, m in zip(axes, models):
        x = _tp_pp_axis(ax)
        for mod in ("text", "image", "video", "video-only"):
            d = get(m, mod)
            if d is None:
                continue
            y = per_config(d.load(), "measured_energy")
            ok = ~np.isnan(y)
            ax.plot(x[ok], y[ok], marker="s", markersize=8, linewidth=2, color=colors[mod],
                    label={"text": "Text", "image": "Image+text", "video": "Video+text",
                           "video-only": "Video only"}[mod], markeredgecolor="white", markeredgewidth=0.8)
            rows += [{"model": m, "modality": mod, "tp": t, "pp": p, "energy": v} for (t, p), v in zip(ORDER, y)]
        ax.set_xticklabels([f"({t}, {p})" for t, p in ORDER], rotation=30, ha="right")
        ax.set_title(m, fontsize=LABEL_SIZE)
        ax.set_yscale("log")   # video-only draws ~5x the energy of text
        style(ax, XLABEL, "Average Energy per Request (J)")
    handles, labels = axes[0].get_legend_handles_labels()   # Qwen2-VL has every modality
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, labels, fontsize=LEGEND_SIZE, ncol=len(labels), loc="lower center", frameon=False)
    fig._laid_out = True
    save(fig, out, "modality_energy")
    r = pd.DataFrame(rows).pivot_table(index=["model", "modality"], columns=["tp", "pp"], values="energy")
    r.to_csv(os.path.join(out, "modality_energy.csv"))
    return r


MAVERIQ_FP16 = (0.9214281439781189, 0.8020966649055481, 1.0621229410171509, 0.2586258053779602)   # published


def _maveriq_design(t, p, e):
    a, b, c, d = e
    return np.stack([1 / (t * p ** a), (t - 1) ** b / p ** c, (p - 1) ** d], axis=1)


def fits(out):
    """How the two analytical forms capture the TP/PP patterns of Fig. tp_pp (numbers for Sec. V.B.2).

    MaverIQ latency, L = L_ref/(t p^a) + (t-1)^b / p^c O_TP + (p-1)^d O_PP: exponents shared by
    all models, L_ref, O_TP, O_PP solved per scenario (non-negative least squares over its six
    layouts, as MaverIQ solves them from its reference layouts). EnergyLens: Eq. (unified-tp-pp)
    fitted per model on all its rows; its TP/PP exponents and per-GPU overheads are reported."""
    from scipy.optimize import minimize, nnls
    from compare_cross_model import fit_unified, unified_predict
    ds = [d for d in _by_family(select("text")) if d.model not in ("Qwen2-VL-7B", "LLaVA-1.5-7B")]
    groups = []
    for d in ds:
        a = d.load().groupby(SCENARIO + ["tp_degree", "pp_degree"]).measured_latency.mean().reset_index()
        for _, g in a.groupby(SCENARIO):
            if len(g) == len(ORDER):
                groups.append((d.model, g.tp_degree.to_numpy(float), g.pp_degree.to_numpy(float),
                               g.measured_latency.to_numpy(float)))

    def predict(e):
        res = []
        for m, t, p, y in groups:
            A = _maveriq_design(t, p, e)
            w = 1 / y                                    # relative error
            coef, _ = nnls(A * w[:, None], np.ones_like(y))
            res.append((m, A @ coef, y, coef))
        return res

    def loss(e):
        return float(np.mean(np.concatenate([np.abs(pr - y) / y for _, pr, y, _ in predict(e)])))
    best = min((minimize(loss, x0, method="L-BFGS-B", bounds=[(0, 3)] * 4)
                for x0 in (MAVERIQ_FP16, (0.1, 1, 1, 1), (0.5, 0.5, 0.5, 0.5))), key=lambda r: r.fun)
    rows = []
    for label, e in (("MaverIQ, published fp16 exponents", MAVERIQ_FP16), ("MaverIQ, exponents fitted here", best.x)):
        r = predict(e)
        per = pd.DataFrame([{"model": m, "ape": a} for m, pr, y, _ in r for a in np.abs(pr - y) / y])
        share_pp = np.median([c[2] * 1 / max(c[0], 1e-9) for *_, c in r])
        rows.append({"form": label, "alpha": e[0], "beta": e[1], "gamma": e[2], "delta": e[3],
                     "latency_mape": 100 * per.ape.mean(), "scenarios": len(r),
                     "models": per.model.nunique(), "median_O_PP_over_L_ref": share_pp})
    lat = pd.DataFrame(rows)
    lat.to_csv(os.path.join(out, "fit_maveriq_latency.csv"), index=False)

    names = ["alpha_p", "beta_p", "gamma1_p", "gamma2_p", "eps_p", "alpha_d", "beta_d", "gamma1_d",
             "gamma2_d", "eps_d", "delta1", "delta2"]
    erows = []
    for d in ds:
        df = d.load()
        th = fit_unified(df, restarts=10)
        pred = unified_predict(th, df)
        erows.append({"model": d.model, "family": d.family, **dict(zip(names, th)),
                      "mape": 100 * np.mean(np.abs(pred - df.measured_energy) / df.measured_energy)})
    en = pd.DataFrame(erows)
    en.to_csv(os.path.join(out, "fit_energylens_tp_pp.csv"), index=False)
    print(lat.round(3).to_string())
    return en[["model", "family", "gamma1_p", "gamma2_p", "gamma1_d", "gamma2_d", "delta1", "delta2", "beta_d", "mape"]]

LAYOUT_COLOR = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a"}          # by GPU count, as in the live-serving figure
STRUCT_LAYOUTS = [(1, 1), (2, 1), (4, 1), (1, 4)]


def _layout_kw(tp, pp):
    ls, mk = ("-", "o") if pp == 1 else ("--", "s")
    return dict(color=LAYOUT_COLOR[tp * pp], ls=ls, marker=mk)


def _struct_handles():
    from matplotlib.lines import Line2D
    h = [Line2D([], [], **_layout_kw(tp, pp), ms=8, mec="white", lw=1.6, label=f"TP{tp}$\\times$PP{pp}")
         for tp, pp in STRUCT_LAYOUTS]
    return h + [Line2D([], [], color="#555555", marker="o", ls="none", ms=8, label="measured"),
                Line2D([], [], color="#555555", ls="-", lw=1.6, label="fitted template")][:0] + \
           [Line2D([], [], color="#555555", ls="none", marker="o", ms=0, label="points: measured; lines: fitted")]


def structure(out, model="Llama-3.1-8B"):
    """Measured energy (and latency) against the three axes of Eq. (unified-tp-pp), with the fitted
    template overlaid: output tokens and batch size from the standard grid, input tokens from the
    long-prompt grid at a fixed batch of 16 requests. Writes structure_energy (energy row, for the
    body) and structure_full (energy and latency rows, for the appendix)."""
    from compare_cross_model import fit_unified, unified_predict, prepare_df
    grid = get(model, "text").load()
    inlen_path = os.path.join(NVIDIA, "results_zeus_inlen", "run_text_only_Llama-3.1-8B-Instruct_fp16", "in_all",
                              "profiling_metrics.csv")
    inl = prepare_df(pd.read_csv(inlen_path), inlen_path)
    th_g, th_i = fit_unified(grid, restarts=10), fit_unified(inl, restarts=10)
    grid = grid.assign(pred=unified_predict(th_g, grid))
    inl = inl.assign(pred=unified_predict(th_i, inl), inreq=inl.total_input / inl.batch_size)
    bins = np.unique(np.round(np.geomspace(1, 512, 12)))
    rows = []

    def panels(axes, metric):
        a, b, c = axes
        for tp, pp in STRUCT_LAYOUTS:
            kw = _layout_kw(tp, pp)
            lab = f"TP{tp}$\\times$PP{pp}"
            g = grid[(grid.tp_degree == tp) & (grid.pp_degree == pp)]
            # (a) output tokens: the same prompts and batches at every output limit
            o = g.groupby("max_tokens").agg(x=("decode_tokens", "mean"), y=(metric, "mean"), f=("pred", "mean"))
            a.plot(o.x, o.y, marker=kw["marker"], color=kw["color"], ls="none", ms=8, mec="white", label=lab)
            if metric == "measured_energy":
                a.plot(o.x, o.f, ls=kw["ls"], color=kw["color"], lw=1.6)
            # (b) input tokens per request at a fixed batch of 16
            i = inl[(inl.tp_degree == tp) & (inl.pp_degree == pp) & (inl.batch_size == 16) & (inl.max_tokens == 256)]
            if len(i):
                lv = i.assign(l=i.inreq.apply(lambda v: min((128, 512, 2048, 4096), key=lambda L: abs(np.log(v / L))))).groupby("l").agg(x=("inreq", "mean"), y=(metric, "mean"))
                b.plot(lv.x, lv.y, marker=kw["marker"], color=kw["color"], ls="none", ms=8, mec="white")
                if metric == "measured_energy":
                    xs = np.linspace(128, 4096, 40)
                    syn = pd.DataFrame({"batch_size": 16, "tp_degree": tp, "pp_degree": pp, "max_tokens": 256,
                                        "decode_tokens": i.decode_tokens.mean(), "total_input": xs * 16,
                                        "max_num_seqs": i.max_num_seqs.iloc[0]})
                    b.plot(xs, unified_predict(th_i, syn), ls=kw["ls"], color=kw["color"], lw=1.6)
            # (c) batch size, log-log
            k = _log_binned(g, metric, bins)
            c.plot(k.x, k.y, marker=kw["marker"], color=kw["color"], ls="none", ms=8, mec="white")
            if metric == "measured_energy":
                kp = _log_binned(g, "pred", bins)
                c.plot(kp.x, kp.y, ls=kw["ls"], color=kw["color"], lw=1.6)
                rows.extend({"layout": lab, "axis": ax_, "x": float(x), "measured": float(y)}
                            for ax_, t in (("output", o), ("batch", k)) for x, y in zip(t.x, t.y))
        ylab = "Energy per request (J)" if metric == "measured_energy" else "Batch latency (s)"
        style(a, "Generated tokens (longest output)", ylab)
        style(b, "Input tokens per request", ylab)
        style(c, "Batch size (requests)", ylab)
        c.set_xscale("log"); c.set_yscale("log")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    panels(axes, "measured_energy")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.legend(handles=_struct_handles(), fontsize=LEGEND_SIZE, ncol=5, loc="lower center", frameon=False)
    fig._laid_out = True
    save(fig, out, "structure_energy")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    panels(axes[0], "measured_energy")
    panels(axes[1], "measured_latency")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(handles=_struct_handles(), fontsize=LEGEND_SIZE, ncol=5, loc="lower center", frameon=False)
    fig._laid_out = True
    save(fig, out, "structure_full")
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(out, "structure.csv"), index=False)
    return r




POWER_MODELS = [("Qwen2-VL-7B", "text"), ("Qwen2-VL-7B", "image"), ("Qwen2-VL-7B", "video"),
                ("LLaVA-1.5-7B", "text"), ("LLaVA-1.5-7B", "image"), ("LLaVA-1.5-7B", "video")]
MOD_STYLE = {"text": ("o", "-"), "image": ("s", "--"), "video": ("^", ":")}
A100_TDP = 400


def power(out, pareto_model="Qwen2.5-14B"):
    """ML.ENERGY-style pair: (a) GPU power against batch size on one A100 for the vision-language
    models per input type, with the TDP; (b) latency against energy per TP/PP layout of one model,
    each normalised to the best layout of the same scenario (median over scenarios), labelled with
    the layout's power per GPU: the fastest layout draws the most power and is not the cheapest."""
    plt.rcParams.update({"axes.titlesize": 19, "axes.labelsize": 18, "xtick.labelsize": 16,
                         "ytick.labelsize": 16, "legend.fontsize": 14, "font.size": 15})
    fig, (a, c, b) = plt.subplots(1, 3, figsize=(17, 5.2))
    bins = np.unique(np.round(np.geomspace(1, 512, 12)))
    colors = {"Qwen2-VL-7B": "#1f77b4", "LLaVA-1.5-7B": "#d62728"}
    rows = []
    for model, mod in POWER_MODELS:
        d = get(model, mod)
        if d is None:
            continue
        df = d.load()
        g = df[(df.tp_degree == 1) & (df.pp_degree == 1)]
        k = _log_binned(g, "gpu_power_w", bins)
        mk, ls = MOD_STYLE[mod]
        a.plot(k.x, k.y, marker=mk, ls=ls, color=colors[model], ms=7, mec="white", lw=1.6,
               label=f"{model}, {mod}")
        rows += [{"panel": "a", "model": model, "modality": mod, "batch": x, "power_w": y} for x, y in zip(k.x, k.y)]
    a.axhline(A100_TDP, color="grey", ls="--", lw=1.2)
    a.text(2, A100_TDP + 6, "A100 TDP (max power draw)", color="grey", fontsize=15, va="bottom")
    a.set_xscale("log", base=2)
    a.set_ylim(0, A100_TDP * 1.15)
    a.set_xlabel("Batch size (requests)")
    a.set_ylabel("Power per GPU (W)")
    a.set_title("(a) One A100, by input type")
    a.legend(fontsize=13, ncol=2, loc="lower right")

    # (b) ML.ENERGY Fig. 5a style: total power of the multi-GPU layouts against the server's TDP
    for model, mod, n, color in (("Qwen2.5-32B", "text", 4, "#2ca02c"), ("InternVL2-76B", "image", 8, "#9467bd")):
        d = get(model, mod)
        if d is None:
            continue
        df = d.load()
        df = df[df.tp_degree * df.pp_degree == n]
        for (tp, pp), g in df.groupby(["tp_degree", "pp_degree"]):
            k = _log_binned(g, "gpu_power_w", bins)
            mk, ls = ("o", "-") if pp == 1 else ("s", "--") if tp == 1 else ("^", "-.")
            c.plot(k.x, k.y, marker=mk, ls=ls, color=color, ms=7, mec="white", lw=1.6,
                   label=f"{model}, TP{tp}$\\times$PP{pp}")
            rows += [{"panel": "b", "model": model, "tp": tp, "pp": pp, "batch": x, "power_w": y}
                     for x, y in zip(k.x, k.y)]
        c.axhline(n * A100_TDP, color=color, ls="--", lw=1.2)
        c.text(2, n * A100_TDP + (40 if n == 8 else -40), f"{n} x A100 TDP (max power draw)", color=color, fontsize=15,
               va="bottom" if n == 8 else "top")
    c.set_xscale("log", base=2)
    c.set_ylim(0, 8 * A100_TDP * 1.12)
    c.set_xlabel("Batch size (requests)")
    c.set_ylabel("Total power draw (W)")
    c.set_title("(b) Multi-GPU layouts vs. server TDP")
    c.legend(fontsize=12.5, loc="center right")

    # (b) ML.ENERGY Fig. 7 style: every configuration (layout x batch size) of one prompt/output length
    df = get(pareto_model, "text").load()
    df = df[(df.max_tokens == 256) & ((df.input_length / df.batch_size).round(-1) == 70)]
    df = df.groupby(["tp_degree", "pp_degree", "batch_size"], as_index=False)[
        ["measured_latency", "measured_energy", "gpu_power_w", "decode_tokens"]].mean()
    df["tpot"] = df.measured_latency / df.decode_tokens
    for (tp, pp), g in df.groupby(["tp_degree", "pp_degree"]):
        kw = _layout_kw(tp, pp)
        b.plot(g.tpot, g.measured_energy, marker=kw["marker"], color=kw["color"], ls="none", ms=8, mec="white",
               label=f"TP{tp}$\\times$PP{pp} ({g.gpu_power_w.mean():.0f} W)")
    srt = df.sort_values(["tpot", "measured_energy"])
    front = srt[srt.measured_energy < srt.measured_energy.cummin().shift(fill_value=np.inf)]
    b.plot(front.tpot, front.measured_energy, color="black", lw=1.6, zorder=3)
    fast, cheap = front.iloc[0], front.iloc[-1]
    for pt in (fast, cheap):        # dashed guides to the axes, as in ML.ENERGY
        b.plot([pt.tpot, pt.tpot], [0, pt.measured_energy], ls="--", color="black", lw=0.9)
        b.plot([0, pt.tpot], [pt.measured_energy, pt.measured_energy], ls="--", color="black", lw=0.9)
    mid = front.iloc[len(front) // 2]
    b.annotate("Pareto frontier", (mid.tpot, mid.measured_energy), xytext=(35, 45), textcoords="offset points",
               fontsize=16, fontweight="bold", arrowprops=dict(arrowstyle="->", lw=1.4))
    b.set_xlim(0, df.tpot.max() * 1.08)
    b.set_ylim(0, df.measured_energy.max() * 1.1)
    b.set_xlabel("Time per output token (s)")
    b.set_ylabel("Energy per request (J)")
    b.set_title(f"(c) Time-energy trade-off, {pareto_model}")
    b.legend(fontsize=13, loc="upper right")
    b.grid(alpha=0.3)
    for r in df.itertuples():
        rows.append({"panel": "c", "model": pareto_model, "tp": r.tp_degree, "pp": r.pp_degree,
                     "batch": r.batch_size, "tpot_s": r.tpot, "energy_j": r.measured_energy,
                     "power_w": r.gpu_power_w})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "power.csv"), index=False)
    save(fig, out, "power")
    return res


FIGURES = {"tp_pp": tp_pp, "optimum": optimum, "batch": batch, "modality": modality, "fits": fits,
           "structure": structure, "power": power}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="results_figures/characterization")
    ap.add_argument("--only", nargs="+", choices=list(FIGURES), default=list(FIGURES))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name in args.only:
        print(f"\n== {name}")
        r = FIGURES[name](args.out)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(r.round(3).to_string())


if __name__ == "__main__":
    main()
