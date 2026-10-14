"""Profiling cost vs accuracy: EnergyLens, MaverIQ (paper protocol), Gradient Boosting, brute force.

Every method is charged like the real profiler (simulator.Simulator): one engine
reload per (TP, PP, budget) visited plus each batch's latency; GPU-hours weight
both by the engine's GPU count. Accuracy is energy MAPE and configuration
regret on the simulator's held-out scenarios.

  brute force           every row of the grid (AlpaServe/DynamoLLM-style lookup)
  EnergyLens random n   n random rows
  EnergyLens D-opt      doptimal_cost, 50 rows
  Gradient Boosting     50 random rows
  MaverIQ (12 rows)     the authors' closed form (MaverIQ Sec. 5.3/5.5, their
                        tools/estimator): 3 cheapest reference layouts x 2 output
                        lengths x 2 batch sizes; per layout, latency is linear in
                        output length (TTFT + out x TPOT) and in batch size (Eq. 6);
                        with alpha..delta fixed, base latency and TP/PP overheads
                        solve a 3x3 linear system (Eq. 2/3), clamped at 0. Exponents:
                        the authors' published fp16 values, or fitted their way
                        (MAPE) on the other models. Energy = latency x per-GPU power
                        of the reference rows x GPUs / batch size
  EnergyLens same rows  EnergyLens fitted on MaverIQ's 12 rows (identical cost)
  MaverIQ fingerprint   the protocol MaverIQ actually uses (Sec. 5.4): the same
                        reference rows measured on the model truncated to PP and
                        PP+1 layers (fingerprint_profiler.py) and extrapolated per
                        layer to the full model, then the same closed form. Power
                        is either what the fingerprints draw ("fingerprint power")
                        or extrapolated per layer like latency ("per-layer energy").
                        Charged with the fingerprints' own latencies and start-ups.
                        Only for models with fingerprint_results/<model>/.

latency_mape is the closed form's latency error alone (MaverIQ rows only).

    python profiling_cost.py        # -> results/profiling_cost.csv, profiling_cost.png
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from simulator import DATASETS, RELOAD_S, ENGINE, ZEUS, Simulator, load, ranking_metrics
from strategies import STRATEGIES
from compare_cross_model import (fit_unified, unified_predict, get_ml_features, prepare_df, drop_invalid_rows,
                                 LC_LABEL_SIZE, LC_TICK_SIZE, LC_LEGEND_SIZE, LC_COLORS)

HERE = os.path.dirname(os.path.abspath(__file__))


# ── MaverIQ closed form (authors' tools/estimator, generalised to any 3 reference layouts) ──
PUBLISHED_FP16 = (0.9214281439781189, 0.8020966649055481, 1.0621229410171509, 0.2586258053779602)


def _design(layouts, exps):
    a, b, c, d = exps
    return np.array([[1 / (t * p ** a), (t - 1) ** b / p ** c, (p - 1) ** d] for t, p in layouts])


def reference_latency(ref):
    """Per reference layout: latency(out, bs) linear in output length and batch size."""
    fns = {}
    for (t, p), g in ref.groupby(["tp_degree", "pp_degree"]):
        per_bs = {}
        for bs, h in g.groupby("batch_size"):
            h = h.sort_values("decode_tokens")
            o, l = h.decode_tokens.to_numpy(float), h.measured_latency.to_numpy(float)
            tpot = (l[-1] - l[0]) / (o[-1] - o[0]) if len(h) > 1 and o[-1] > o[0] else 0.0
            per_bs[bs] = (l[0] - o[0] * tpot, tpot)
        (b1, (f1, s1)), (b2, (f2, s2)) = sorted(per_bs.items())[0], sorted(per_bs.items())[-1]

        def lat(out, bs, b1=b1, b2=b2, f1=f1, s1=s1, f2=f2, s2=s2):
            l1, l2 = f1 + out * s1, f2 + out * s2
            slope = np.maximum((l2 - l1) / (b2 - b1), 0) if b2 > b1 else 0.0
            return l1 + np.maximum(bs - b1, 0) * slope   # like the authors: never below the first batch size
        fns[(t, p)] = lat
    return fns


def maveriq_latency(ref, test, exps):
    fns = reference_latency(ref)
    layouts = sorted(fns)
    A_inv = np.linalg.inv(_design(layouts, exps))
    out, bs = test.decode_tokens.to_numpy(float), test.batch_size.to_numpy(float)
    L = np.stack([fns[k](out, bs) for k in layouts])            # 3 x n
    base, tpo, ppo = np.maximum(A_inv @ L, 0)
    a, b, c, d = exps
    t, p = test.tp_degree.to_numpy(float), test.pp_degree.to_numpy(float)
    return base / (t * p ** a) + (t - 1) ** b / p ** c * tpo + (p - 1) ** d * ppo


def maveriq_energy_closed_form(ref, test, exps):
    per_gpu = np.median(ref.gpu_power_w / (ref.tp_degree * ref.pp_degree))
    lat = np.maximum(maveriq_latency(ref, test, exps), 1e-9)
    return lat * per_gpu * test.tp_degree.to_numpy() * test.pp_degree.to_numpy() / test.batch_size.to_numpy()


def fitted_exponents(name, seed, n=300):
    """alpha..delta fitted once across the other models, as the authors do (MAPE)."""
    rng = np.random.RandomState(seed)
    sets = []
    for o in DATASETS:
        if o == name:
            continue
        sim = Simulator(o, seed)
        d = load(o)
        sets.append((sim.pool.iloc[reference_rows(sim)], d.iloc[rng.choice(len(d), min(n, len(d)), replace=False)]))

    def loss(e):
        errs = [np.abs(maveriq_latency(r, t, e) - t.measured_latency) / t.measured_latency for r, t in sets]
        return float(np.mean(np.concatenate(errs)))
    best = min((minimize(loss, x0, method="L-BFGS-B", bounds=[(1e-6, 3)] * 4) for x0 in
                [PUBLISHED_FP16, (0.05, 0.75, 0.5, 0.1), (0.5, 0.5, 0.5, 0.5)]), key=lambda r: r.fun)
    return tuple(best.x)


def reference_rows(sim):
    """3 cheapest layouts (fewest GPUs, then TP) at the middle budget: the smallest and the
    largest batch size measured at both the shortest and longest output length (12 rows);
    MaverIQ uses batch sizes 1 and 2, which the grid does not contain."""
    pool = sim.pool.assign(g=sim.pool.tp_degree * sim.pool.pp_degree, idx=range(len(sim.pool)))
    budget = sorted(pool.budget.unique())[len(pool.budget.unique()) // 2]
    layouts = (pool[["tp_degree", "pp_degree", "g"]].drop_duplicates()
               .sort_values(["g", "pp_degree"]).head(3).itertuples(index=False))
    rows = []
    for tp, pp, _ in layouts:
        e = pool[(pool.tp_degree == tp) & (pool.pp_degree == pp) & (pool.budget == budget)]
        lo, hi = e.max_tokens.min(), e.max_tokens.max()
        common = sorted(set(e[e.max_tokens == lo].batch_size) & set(e[e.max_tokens == hi].batch_size))
        common = [common[0], common[-1]]
        for mt in (lo, hi):
            m = e[(e.max_tokens == mt) & e.batch_size.isin(common)]
            rows += m.drop_duplicates("batch_size").idx.tolist()
    return rows


# ── MaverIQ fingerprints (fingerprint_profiler.py): the reference rows measured on the model ──
# ── truncated to PP and PP+1 layers, extrapolated per layer to the full layer count        ──
PUBLISHED_FP16_FINGERPRINT = (0.04908526688814163, 0.6858410239219666, 0.18765105307102203, 0.568636953830719)
FINGERPRINT_RESULTS = os.path.join(HERE, "fingerprint_results")
FINGERPRINT_PAUSE_S = 10.0   # the profiler's pause after each engine, as in RELOAD_S


def _fingerprint_dir(name):
    run = DATASETS[name]                                    # run_text_only_<model>_fp16
    return os.path.join(FINGERPRINT_RESULTS, run[len("run_text_only_"):-len("_fp16")])


def has_fingerprint(name):
    """All 3 reference layouts profiled at both fingerprint sizes (6 engines)."""
    path = os.path.join(_fingerprint_dir(name), "engines.json")
    return os.path.exists(path) and len(pd.read_json(path)) == 6


def fingerprint_reference(name):
    """Reference rows extrapolated from the fingerprints, plus the profiling charge.

    Per (layout, output length, batch) MaverIQ's per-layer TTFT/TPOT solve is exact
    through its 4 points, i.e. at each measured output length latency is linear in
    the layer count: L(N) = L(n1) + (N - n1) (L(n2) - L(n1)). Energy is extrapolated
    the same way (the 'per-layer energy' variant); power = energy x batch / latency."""
    d = _fingerprint_dir(name)
    raw = pd.read_csv(os.path.join(d, "profiling_metrics.csv"))
    for c in ("latency", "power", "energy"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw[raw.energy > 0].dropna(subset=["latency", "power"]).reset_index(drop=True)
    raw = drop_invalid_rows(raw, d)   # before prepare_df, so num_layers below stays aligned
    fp = prepare_df(raw, d).assign(layers=raw.num_layers.astype(int).to_numpy())
    key = ["tp_degree", "pp_degree", "max_tokens", "batch_size"]
    # repeated measurements of a row (fingerprint_profiler --repeats) are averaged, as
    # MaverIQ averages 10 iterations; the cost below still charges every repeat
    fp_all = fp
    fp = fp.groupby(key + ["layers"], as_index=False)[["decode_tokens", "measured_latency",
                                                        "measured_energy", "gpu_power_w"]].mean()
    full = int(pd.read_csv(os.path.join(ZEUS, DATASETS[name], "res_native", "profiling_metrics.csv"),
                           usecols=["num_layers"], nrows=1).num_layers[0])
    rows = []
    for k, g in fp.groupby(key):
        g = g.sort_values("layers")
        if len(g) < 2:
            continue
        a, b = g.iloc[0], g.iloc[-1]
        x = lambda c: a[c] + (full - a.layers) * (b[c] - a[c]) / (b.layers - a.layers)
        rows.append({**dict(zip(key, k)), "decode_tokens": a.decode_tokens,
                     "measured_latency": x("measured_latency"), "measured_energy": x("measured_energy"),
                     "fp_power_w": g.gpu_power_w.mean()})
    ref = pd.DataFrame(rows)
    ref["gpu_power_w"] = ref.measured_energy * ref.batch_size / ref.measured_latency
    eng = pd.read_json(os.path.join(d, "engines.json"))
    g = eng.tp * eng.pp
    load_s = eng.load_s + FINGERPRINT_PAUSE_S
    secs = fp_all.measured_latency.sum() + load_s.sum()
    gpu_s = (fp_all.measured_latency * fp_all.tp_degree * fp_all.pp_degree).sum() + (g * load_s).sum()
    cost = {"rows": len(fp_all), "engines": len(eng), "minutes": secs / 60, "gpu_hours": gpu_s / 3600}
    return ref, cost


def maveriq_energy_fingerprint(ref, test, exps, power):
    """Closed-form latency from the extrapolated reference rows; power either as measured
    on the fingerprints ('fingerprint') or extrapolated per layer ('per-layer energy')."""
    col = "fp_power_w" if power == "fingerprint" else "gpu_power_w"
    return maveriq_energy_closed_form(ref.assign(gpu_power_w=ref[col]), test, exps)


def fitted_exponents_fingerprint(name, seed, n=300):
    """As fitted_exponents, on the other models' fingerprint-extrapolated reference rows."""
    rng = np.random.RandomState(seed)
    sets = [(fingerprint_reference(o)[0], load(o).pipe(lambda d: d.iloc[rng.choice(len(d), min(n, len(d)), replace=False)]))
            for o in DATASETS if o != name and has_fingerprint(o)]

    def loss(e):
        errs = [np.abs(maveriq_latency(r, t, e) - t.measured_latency) / t.measured_latency for r, t in sets]
        return float(np.mean(np.concatenate(errs)))
    best = min((minimize(loss, x0, method="L-BFGS-B", bounds=[(1e-6, 3)] * 4) for x0 in
                [PUBLISHED_FP16_FINGERPRINT, PUBLISHED_FP16, (0.5, 0.5, 0.5, 0.5)]), key=lambda r: r.fun)
    return tuple(best.x)


def gb_predict(train, test):
    sc = StandardScaler().fit(get_ml_features(train))
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(sc.transform(get_ml_features(train)), train.measured_energy)
    return gb.predict(sc.transform(get_ml_features(test)))


def charge(df, reload_s):
    """Minutes and GPU-hours to profile df's rows, one reload per engine."""
    g = df.tp_degree * df.pp_degree
    engines = df.assign(g=g).drop_duplicates(ENGINE)
    secs = df.measured_latency.sum() + len(engines) * reload_s
    gpu_s = (df.measured_latency * g).sum() + (engines.g * reload_s).sum()
    return {"engines": len(engines), "minutes": secs / 60, "gpu_hours": gpu_s / 3600}


def score(pred, test):
    y = test.measured_energy.to_numpy(float)
    return {"mape": 100 * np.mean(np.abs(pred - y) / y), **ranking_metrics(test, pred)}


def run_one(task):
    name, seed, sizes = task
    reload_s, out = RELOAD_S[name], []
    add = lambda method, rows_df, pred, test: out.append(
        {"dataset": name, "seed": seed, "method": method, "rows": len(rows_df),
         **charge(rows_df, reload_s), **score(pred, test)})

    full = load(name)
    out.append({"dataset": name, "seed": seed, "method": "brute force (full grid)", "rows": len(full),
                **charge(full, reload_s), "mape": 0.0, "regret": 0.0, "top1": 100.0, "pairwise": 100.0})

    for n in sizes:
        sim = Simulator(name, seed)
        rows = list(np.random.RandomState(seed).choice(sim.available(), n, replace=False))
        tr = sim.pool.iloc[rows]
        add(f"EnergyLens, random {n}", tr, unified_predict(fit_unified(tr), sim.test), sim.test)
        if n == 50:
            add("Gradient Boosting, random 50", tr, gb_predict(tr, sim.test), sim.test)

    sim = Simulator(name, seed)
    rng = np.random.RandomState(seed)
    for rows in STRATEGIES["doptimal_cost"](sim, 50, rng):
        sim.measure(rows[: 50 - len(sim.measured)])
        if len(sim.measured) >= 50:
            break
    tr = sim.observed()
    add("EnergyLens, D-optimal 50", tr, unified_predict(fit_unified(tr), sim.test), sim.test)

    sim = Simulator(name, seed)
    tr = sim.pool.iloc[reference_rows(sim)]
    lat_mape = lambda ref, e: 100 * np.mean(np.abs(maveriq_latency(ref, sim.test, e) - sim.test.measured_latency)
                                            / sim.test.measured_latency)
    for label, exps in (("published exponents", PUBLISHED_FP16),
                        ("exponents fit on other models", fitted_exponents(name, seed))):
        add(f"MaverIQ, {label} (12 rows)", tr, maveriq_energy_closed_form(tr, sim.test, exps), sim.test)
        out[-1]["latency_mape"] = lat_mape(tr, exps)
    add("EnergyLens, same 12 rows", tr, unified_predict(fit_unified(tr), sim.test), sim.test)

    if has_fingerprint(name):
        ref, cost = fingerprint_reference(name)
        for label, exps in (("published fingerprint exponents", PUBLISHED_FP16_FINGERPRINT),
                            ("exponents fit on other models", fitted_exponents_fingerprint(name, seed))):
            for power in ("fingerprint", "per-layer energy"):
                pred = maveriq_energy_fingerprint(ref, sim.test, exps, power)
                out.append({"dataset": name, "seed": seed, "method": f"MaverIQ fingerprint, {label}, {power} power",
                            **cost, **score(pred, sim.test), "latency_mape": lat_mape(ref, exps)})
    return out


def plot(summary, path):
    """Random rows vs D-optimal rows for EnergyLens: energy MAPE and Top-1 selection
    accuracy against profiling cost. The other methods are reported in the text."""
    blue = LC_COLORS["EnergyLens (Ours)"]
    rnd = summary[summary.index.str.startswith("EnergyLens, random")].sort_values("gpu_hours")
    dopt = summary.loc["EnergyLens, D-optimal 50"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, label in [(axes[0], "mape", "Energy MAPE (%)"),
                           (axes[1], "top1", "Top-1 accuracy (%)")]:
        # means over the models; the spread across models is in the text (it dwarfs the
        # differences on the regret panel)
        ax.plot(rnd.gpu_hours, rnd[col], "-o", color="#888888", ms=9, mec="white", mew=1, lw=2,
                label="random rows", zorder=3)
        for m, r in rnd.iterrows():
            ax.annotate(f"n={m.split()[-1]}", (r.gpu_hours, r[col]), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=LC_TICK_SIZE - 3)
        ax.plot(dopt.gpu_hours, dopt[col], "*", color=blue, ms=20, mec="white", mew=1,
                label="D-optimal, n=50", zorder=4)
        ax.annotate("n=50", (dopt.gpu_hours, dopt[col]), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=LC_TICK_SIZE - 3, color=blue)
        ax.set_xscale("log")
        ticks = [0.5, 1, 2]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlim(0.42, 2.4)
        vals = summary.loc[list(rnd.index) + ["EnergyLens, D-optimal 50"], col]
        if col == "top1":       # accuracy: zoom on the range the points span
            ax.set_ylim(vals.min() - 4, min(100, vals.max() + 4))
        else:
            ax.set_ylim(0, vals.max() * 1.35)
        ax.set_xlabel("Profiling cost (GPU-hours)", fontsize=LC_LABEL_SIZE - 3)
        ax.set_ylabel(label, fontsize=LC_LABEL_SIZE - 3)
        ax.tick_params(labelsize=LC_TICK_SIZE - 3)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_facecolor("#f9f9f9")
    axes[0].legend(fontsize=LC_LEGEND_SIZE - 2, framealpha=0.9, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def replot(path):
    r = pd.read_csv(f"{path}.csv")
    per_ds = r.groupby(["method", "dataset"])[["gpu_hours", "mape", "regret", "top1"]].mean()
    summary = per_ds.groupby("method").mean()
    for c in ("mape", "regret"):
        summary[f"{c}_std"] = per_ds[c].groupby("method").std()
    plot(summary.sort_values("gpu_hours"), path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--sizes", type=int, nargs="+", default=[25, 50, 100])
    ap.add_argument("--output", default=os.path.join(HERE, "results", "profiling_cost"))
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from the saved CSV")
    args = ap.parse_args()
    if args.plot_only:
        return replot(args.output)

    tasks = [(d, s, args.sizes) for d in args.datasets for s in args.seeds]
    with mp.Pool(min(len(tasks), os.cpu_count())) as pool:
        r = pd.DataFrame([row for rows in pool.map(run_one, tasks) for row in rows])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    r.to_csv(f"{args.output}.csv", index=False)

    brute = r[r.method.str.startswith("brute")].set_index(["dataset", "seed"]).gpu_hours
    r["cost_pct"] = 100 * r.gpu_hours / r.set_index(["dataset", "seed"]).index.map(brute).to_numpy()
    per_ds = r.groupby(["method", "dataset"])[["rows", "engines", "minutes", "gpu_hours", "cost_pct",
                                              "mape", "latency_mape", "top1", "regret"]].mean()
    summary = per_ds.groupby("method").mean()
    for c in ("mape", "regret"):
        summary[f"{c}_std"] = per_ds[c].groupby("method").std()
    summary = summary.sort_values("gpu_hours")
    print(f"Mean over {r.dataset.nunique()} models x {r.seed.nunique()} seeds "
          f"(cost_pct = % of the brute-force GPU-hours):")
    print(summary.drop(columns=["mape_std", "regret_std"]).round(2).to_string())
    fp_sets = sorted(r[r.method.str.startswith("MaverIQ fingerprint")].dataset.unique())
    if 0 < len(fp_sets) < r.dataset.nunique():
        sub = per_ds[per_ds.index.get_level_values("dataset").isin(fp_sets)].groupby("method").mean()
        print(f"\nOnly the {len(fp_sets)} models with fingerprints ({', '.join(fp_sets)}):")
        print(sub.sort_values("gpu_hours").round(2).to_string())
    plot(summary, args.output)
    print(f"\nSaved -> {args.output}.csv / .png / .pdf")


if __name__ == "__main__":
    main()
