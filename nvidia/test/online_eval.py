#!/usr/bin/env python3
"""Predict the energy of online-replay windows from offline profiling.

    python test/online_eval.py \
        --online results_zeus_online/run_online_Llama-3.1-8B-Instruct_fp16 \
        --grid results_zeus_inlen/run_text_only_Llama-3.1-8B-Instruct_fp16/in_all/profiling_metrics.csv \
        --trace results_zeus_trace/run_trace_Llama-3.1-8B-Instruct_fp16/profiling_metrics.csv
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pairwise_ranking import (prepare_df, fit_unified, unified_predict, fit_maveriq,  # noqa: E402
                              fit_gb, fit_latency, _SKLEARN)
from compare_cross_model import drop_invalid_rows  # noqa: E402

MAX_SEQS = 64



KNOWN, PREDICTED = 'SLO, known feasibility', 'SLO, predicted feasibility'


def fewest_gpus(x):
    return x.sort_values(['gpus', 'pp_degree']).index[0]


def evaluate_slo(agg, methods, ks, keys=('max_tokens', 'batch_size'), with_rule=False):
    """Lowest energy under a latency limit (k x the fastest configuration), with known or
    predicted feasibility. agg: one row per (scenario, config) with lat (measured), plat
    (predicted), gpus, measured_energy and p_<method> energy predictions."""
    out = []
    for _, x in agg.groupby(list(keys)):
        if len(x) < 2:
            continue
        for k in ks:
            slo = k * x.lat.min()
            best = x.measured_energy[x.lat <= slo].min()
            binds = float(x.lat[x.measured_energy.idxmin()] > slo)
            for mode in (KNOWN, PREDICTED):
                lat = x.lat if mode == KNOWN else x.plat
                ok = lat <= slo
                c = x[ok] if ok.any() else x.loc[[lat.idxmin()]]
                picks = {m: (c[f'p_{m}'].idxmin() if c[f'p_{m}'].notna().any() else lat.idxmin()) for m in methods}
                if with_rule:
                    picks['fewest GPUs, then TP'] = fewest_gpus(c)
                picks['fastest'] = lat.idxmin()
                for m, i in picks.items():
                    viol = bool(x.lat[i] > slo)
                    e = x.measured_energy[i]
                    out.append({'setting': mode, 'k': k, 'method': m, 'top1': float(not viol and e == best),
                                'regret': np.nan if viol else (e - best) / best * 100,
                                'violation': float(viol), 'binds': binds})
    return out

def load_offline(path):
    raw = pd.read_csv(path)
    raw = raw[pd.to_numeric(raw["energy"], errors="coerce") > 0].reset_index(drop=True)
    return prepare_df(drop_invalid_rows(raw, path), path)


def pseudo_rows(tp, pp, b, ins, outs, budget=8192, seqs=MAX_SEQS):
    """Prediction rows: request r as one of a batch of round(b) requests like it."""
    bs = max(int(round(b)), 1)
    raw = pd.DataFrame({"latency": 1.0, "energy": 1.0, "power": 1.0,
                        "tensor_parallel_size": tp, "pipeline_parallel_size": pp,
                        "batch_size": bs, "input_length": np.asarray(ins) * bs,
                        "max_tokens": np.asarray(outs), "max_generated_tokens": np.asarray(outs),
                        "max_num_batched_tokens": budget, "max_num_seqs": seqs})
    return prepare_df(raw, "online")


def fit_all(train):
    lat = fit_latency(train)
    theta = fit_unified(train)
    _, mav_lat = fit_maveriq(train)
    # MaverIQ multiplies its latency by power; online the window's power is unknown,
    # so it gets each layout's median power over its training rows.
    p_med = train.groupby(["tp_degree", "pp_degree"])["gpu_power_w"].median().to_dict()
    per_req = {"EnergyLens": lambda d: unified_predict(theta, d),
               "MaverIQ": lambda d: mav_lat(d) * d.apply(
                   lambda r: p_med.get((r.tp_degree, r.pp_degree), np.nan), axis=1).values
               / np.maximum(d["batch_size"].values, 1)}
    if _SKLEARN:
        gb = fit_gb(train)
        per_req["Gradient Boosting"] = gb
    # latency model each method's SLO check uses (MaverIQ: its own; others: fit_latency)
    lat_of = {m: (mav_lat if m == "MaverIQ" else lat) for m in per_req}
    return lat, per_req, lat_of


def predicted_load(lat, tp, pp, n, span, ins, outs, budget=8192, seqs=MAX_SEQS):
    """Little's law with the batch latency model, b = lambda * L(b), plus a fluid
    queue: a layout serves at most MAX_SEQS requests per L(MAX_SEQS) seconds; when
    requests arrive faster, the backlog left at the end of the arrivals drains at
    that rate, adding to the duration and to the latency of the late requests.
    Returns (load b, duration D, p90 request latency)."""
    lam = n / max(span, 1e-3)
    L_at = lambda b: float(np.median(lat(pseudo_rows(tp, pp, b, ins, outs, budget, seqs))))
    L_full = L_at(seqs)
    mu = seqs / L_full                              # requests per second at full batch
    if lam >= mu:
        backlog = n - mu * span
        return float(seqs), span + backlog / mu + L_full, L_full + 0.9 * backlog / mu
    b = 1.0
    for _ in range(50):
        b_new = min(lam * L_at(max(b, 1.0)), seqs)
        if abs(b_new - b) < 1e-3:
            break
        b = 0.5 * b + 0.5 * b_new
    L = L_at(max(b, 1.0))
    return b, span + L, L


def predict_windows(online, reqs, lat, per_req, load, lat_of):
    out = []
    for r in online.itertuples():
        key = f"tp{r.tensor_parallel_size}_pp{r.pipeline_parallel_size}"
        full = f"{key}_s{r.max_num_seqs}_b{r.max_num_batched_tokens}"
        rq = reqs.get((full, r.window_id)) or reqs[(key, r.window_id)]
        bud, sq = int(r.max_num_batched_tokens), int(r.max_num_seqs)
        ins = [q["input"] for q in rq]
        outs = [q["output"] for q in rq]
        if load == "measured":
            b, D = r.mean_in_flight, r.duration_s
        else:
            b, D, _ = predicted_load(lat, r.tensor_parallel_size, r.pipeline_parallel_size,
                                     len(rq), r.arrival_span_s, ins, outs, bud, sq)
        d = pseudo_rows(r.tensor_parallel_size, r.pipeline_parallel_size, b, ins, outs, bud, sq)
        idle = r.idle_power * D * max(0.0, 1.0 - b)
        row = {"window_id": r.window_id, "trace": r.trace, "rate": r.rate,
               "tp": r.tensor_parallel_size, "pp": r.pipeline_parallel_size,
               "seqs": sq, "budget": bud, "hour": getattr(r, "hour", np.nan),
               "energy": r.energy, "load": load, "b": b, "b_measured": r.mean_in_flight,
               "p90_e2e_s": r.p90_e2e_s}
        for name, f in per_req.items():
            row[name] = float(np.sum(f(d))) + idle
            # predicted p90 latency for the SLO check, with the method's latency model
            row[f"lat_{name}"] = predicted_load(lat_of[name], r.tensor_parallel_size, r.pipeline_parallel_size,
                                                len(rq), r.arrival_span_s, ins, outs, bud, sq)[2]
        out.append(row)
    return pd.DataFrame(out)


def ranking(pred, methods):
    res = {m: {"pairwise": [], "top1": [], "regret": []} for m in methods}
    for _, g in pred.groupby("window_id"):
        if len(g) < 2:
            continue
        y = g["energy"].to_numpy()
        for m in methods:
            p = g[m].to_numpy()
            dy, dp = y[:, None] - y[None, :], p[:, None] - p[None, :]
            valid = np.triu(np.abs(dy) > 1e-9, 1)
            res[m]["pairwise"].append(((dy * dp) > 0)[valid].mean() if valid.any() else 1.0)
            res[m]["top1"].append(p.argmin() == y.argmin())
            res[m]["regret"].append((y[p.argmin()] - y.min()) / y.min() * 100)
    return {m: {k: 100 * np.mean(v) if k != "regret" else np.mean(v) for k, v in r.items()}
            for m, r in res.items()}


def slo(pred, methods, ks):
    """MaverIQ-style SLO selection per window: the lowest-energy layout whose p90
    request latency is within k x the fastest layout's, with feasibility known
    (measured p90) or predicted (each method's latency model + fluid queue)."""
    rows = []
    for m in methods:
        agg = pred.assign(lat=pred["p90_e2e_s"], plat=pred[f"lat_{m}"], gpus=pred["tp"] * pred["pp"],
                          measured_energy=pred["energy"], tp_degree=pred["tp"], pp_degree=pred["pp"],
                          **{f"p_{m}": pred[m]})
        for r in evaluate_slo(agg, [m], ks, keys=["window_id"], with_rule=(m == methods[0])):
            if r["method"] in (m, "fewest GPUs, then TP", "fastest") or "latency x GPUs" in r["method"]:
                rows.append(r)
    out = pd.DataFrame(rows).drop_duplicates()
    return (out.groupby(["setting", "k", "method"])[["top1", "regret", "violation", "binds"]]
            .agg(lambda v: np.nanmean(v)).reset_index())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--online", required=True, help="results_zeus_online/run_online_<model>_<tag>")
    ap.add_argument("--grid", help="offline grid CSV (input-length grid where available)")
    ap.add_argument("--trace", help="offline trace-batch CSV of the same model")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = args.out or os.path.join("results_figures", "online_" + os.path.basename(args.online.rstrip("/")))
    os.makedirs(out_dir, exist_ok=True)

    online = pd.read_csv(os.path.join(args.online, "profiling_metrics.csv"))
    reqs = {}
    req_path = os.path.join(args.online, "requests.jsonl")
    if not os.path.exists(req_path):                      # the released data ships it gzipped
        import gzip
        req_file = gzip.open(req_path + ".gz", "rt")
    else:
        req_file = open(req_path)
    for line in req_file:
        j = json.loads(line)
        reqs[(j["cfg"], j["window_id"])] = j["requests"]
    rule = online.assign(g=online.tensor_parallel_size * online.pipeline_parallel_size)
    online["J_per_req"] = online["energy"] / online["n_requests"]
    print(online.groupby(["trace", "rate", "tensor_parallel_size", "pipeline_parallel_size"])
          [["J_per_req", "mean_in_flight", "duration_s", "idle_power"]].mean().round(1).to_string())

    sets = {}
    if args.grid:
        sets["grid"] = load_offline(args.grid)
    if args.trace:
        sets["offline trace"] = load_offline(args.trace)
    if args.grid and args.trace:
        sets["grid + offline trace"] = pd.concat([sets["grid"], sets["offline trace"]], ignore_index=True)

    rows, preds, slos = [], [], []
    for name, train in sets.items():
        lat, per_req, lat_of = fit_all(train)
        for load in ("measured", "predicted"):
            p = predict_windows(online, reqs, lat, per_req, load, lat_of)
            sl = slo(p, list(per_req), [1.5, 2, 3, 5])
            sl.insert(0, "load", load)
            sl.insert(0, "train", name)
            slos.append(sl)
            p["train"] = name
            preds.append(p)
            methods = list(per_req)
            rk = ranking(p, methods)
            for m in methods:
                rows.append({"train": name, "load": load, "method": m,
                             "mape": 100 * np.mean(np.abs(p[m] - p["energy"]) / p["energy"]),
                             **rk[m]})
    # the no-model rule: fewest GPUs, then TP
    r = rule.assign(score=rule["g"] * 10 + rule["pipeline_parallel_size"])
    rk = ranking(r.assign(rule=r["score"].astype(float)), ["rule"])["rule"]
    rows.append({"train": "-", "load": "-", "method": "fewest GPUs, then TP", "mape": np.nan, **rk})

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "online_eval_summary.csv"), index=False)
    pd.concat(preds).to_csv(os.path.join(out_dir, "online_eval_windows.csv"), index=False)
    print(summary.round(2).to_string(index=False))
    slo_df = pd.concat(slos)
    slo_df[["top1", "violation", "binds"]] *= 100
    slo_df.to_csv(os.path.join(out_dir, "online_slo_summary.csv"), index=False)
    print(slo_df.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
