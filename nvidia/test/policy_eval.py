#!/usr/bin/env python3
"""End-to-end layout selection on live-replayed Azure traffic (Test 1).

Every layout replays the same windows (source/online_trace.py), so each window has a
measured energy and p90 request latency per layout, and a selection policy can be scored
exactly: it picks one layout per window and is charged that layout's measured energy and
latency. Latency target per window: p90 within k x the fastest layout's p90.

Policies
  oracle                  lowest measured energy among layouts meeting the target
  EnergyLens / MaverIQ / GB, two variants:
    measured feasibility  told which layouts meet the target, rank them by predicted energy
    predicted feasibility target checked with the method's own latency prediction
  fewest GPUs, then TP    no model; with measured feasibility: first compliant layout
  fastest                 lowest measured p90 (always compliant)
Reported per k: energy over all windows relative to the oracle, share of windows
meeting the target, and the same split by arrival rate.

    python test/policy_eval.py results_figures/online_run_online_*  [--train grid --load predicted]
"""
import argparse
import os

import numpy as np
import pandas as pd

METHODS = ["EnergyLens", "MaverIQ", "Gradient Boosting"]


def rule_order(g):
    """Fewest GPUs, then TP; among a layout's serving settings the default-like one
    (concurrency limit 128, token budget 8192) first."""
    g = g.assign(n=g.tp * g.pp)
    if "seqs" in g:
        g = g.assign(_d=~((g.seqs == 128) & (g.budget == 8192)))
        return g.sort_values(["n", "pp", "_d"])
    return g.sort_values(["n", "pp"])


def label(r):
    return f"({r.tp},{r.pp})" + (f" s{int(r.seqs)} b{int(r.budget)}" if "seqs" in r and r.seqs == r.seqs else "")


def static_best(d, k):
    """One configuration for the whole day, chosen with hindsight: the lowest total energy
    among those meeting the target in the most windows."""
    rows = []
    for model, m in d.groupby("model"):
        keys = ["tp", "pp"] + (["seqs", "budget"] if "seqs" in m else [])
        t = m.assign(target=m.groupby("window_id").p90_e2e_s.transform("min") * k)
        t["meets"] = t.p90_e2e_s <= t.target + 1e-9
        s = t.groupby(keys).agg(meets=("meets", "mean"), energy=("energy", "sum"), n=("window_id", "nunique"))
        s = s[s.n == m.window_id.nunique()]
        best = s.sort_values(["meets", "energy"], ascending=[False, True]).index[0]
        sel = t.set_index(keys).loc[best].reset_index()
        for r in sel.itertuples():
            rows.append({"k": k, "model": model, "window_id": r.window_id, "rate": r.rate, "trace": r.trace,
                         "hour": getattr(r, "hour", np.nan), "policy": "static best (hindsight)",
                         "layout": str(best), "energy": r.energy, "p90": r.p90_e2e_s, "target": r.target,
                         "meets": r.meets})
    return rows


def choose(g, k):
    """One window's choices: {policy: chosen row}."""
    target = k * g.p90_e2e_s.min()
    ok = g.p90_e2e_s <= target + 1e-9
    feas = g[ok]
    out = {"oracle": feas.loc[feas.energy.idxmin()],
           "fastest": g.loc[g.p90_e2e_s.idxmin()],
           "fewest GPUs, then TP": rule_order(g).iloc[0],
           "fewest GPUs, then TP (meas. feas.)": rule_order(feas).iloc[0]}
    for m in METHODS:
        if m not in g:
            continue
        out[f"{m} (meas. feas.)"] = feas.loc[feas[m].idxmin()]
        pf = g[g[f"lat_{m}"] <= k * g[f"lat_{m}"].min() + 1e-9]
        out[f"{m} (pred. feas.)"] = pf.loc[pf[m].idxmin()]
    return out, target


def evaluate(d, ks):
    rows = []
    for k in ks:
        rows += static_best(d, k)
        for (model, wid), g in d.groupby(["model", "window_id"]):
            picks, target = choose(g, k)
            for pol, r in picks.items():
                rows.append({"k": k, "model": model, "window_id": wid, "rate": g.rate.iloc[0],
                             "trace": g.trace.iloc[0], "hour": g["hour"].iloc[0] if "hour" in g else np.nan,
                             "policy": pol, "layout": label(r), "p90": r.p90_e2e_s, "target": target,
                             "energy": r.energy, "oracle": picks["oracle"].energy,
                             "meets": r.p90_e2e_s <= target + 1e-9})
    r = pd.DataFrame(rows)
    o = r[r.policy == "oracle"].set_index(["k", "model", "window_id"]).energy
    r["oracle"] = r.set_index(["k", "model", "window_id"]).index.map(o).to_numpy()
    return r


def summary(r):
    g = r.groupby(["k", "policy"]).agg(energy=("energy", "sum"), oracle=("oracle", "sum"), meets=("meets", "mean"))
    g["extra_energy_%"] = 100 * (g.energy / g.oracle - 1)
    g["meets_%"] = 100 * g.meets
    return g[["extra_energy_%", "meets_%"]].round(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="results_figures/online_run_online_<model> folders")
    ap.add_argument("--train", default="grid", help="training set of the energy models (online_eval.py)")
    ap.add_argument("--load", default="predicted", choices=["measured", "predicted"])
    ap.add_argument("--k", type=float, nargs="+", default=[1.5, 2.0, 3.0])
    ap.add_argument("--out", default="results_figures/policy_eval")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = pd.concat([pd.read_csv(os.path.join(p, "online_eval_windows.csv")).assign(
        model=os.path.basename(p.rstrip("/")).replace("online_run_online_", "")) for p in a.runs])
    d = d[(d.train == a.train) & (d.load == a.load)]
    r = evaluate(d, a.k)
    r.to_csv(os.path.join(a.out, f"policy_windows_{a.train.replace(' ', '_')}_{a.load}.csv"), index=False)
    s = summary(r)
    s.to_csv(os.path.join(a.out, f"policy_summary_{a.train.replace(' ', '_')}_{a.load}.csv"))
    pd.set_option("display.width", 200)
    print(f"train = {a.train}, load = {a.load}, windows = {d.groupby(['model', 'window_id']).ngroups}")
    print(s.unstack(0).to_string())
    print("\nextra energy over the oracle by arrival rate (k = 2):")
    t = r[r.k == 2.0].groupby(["rate", "policy"]).agg(e=("energy", "sum"), o=("oracle", "sum"), m=("meets", "mean"))
    print((100 * (t.e / t.o - 1)).round(1).unstack(0).to_string())
    print("\nlayouts chosen at k = 2 (count over windows):")
    print(r[r.k == 2.0].groupby(["policy", "layout"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
