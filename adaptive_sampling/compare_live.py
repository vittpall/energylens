"""Compare every live run with random 50-row samples drawn from the same reference grid.

Random draws are scored exactly like live runs (fit on the sample, test on every
reference scenario it did not touch) and timed with the costs measured live:
median engine load + 10 s cleanup per engine, recorded batch latency, and the
same one-time cold start.

    python compare_live.py                    # writes live_results/comparison.csv
    python compare_live.py --decode measured  # -> live_results/comparison_measured.csv

--decode measured fits and tests with each batch's measured longest output
(max_generated_tokens) instead of the max_tokens limit, for models whose
reference grid was re-profiled with that column (results_zeus_maxgen/, used in
place of results_zeus/); other models fall back to the limit.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import glob
import json
import multiprocessing as mp

import numpy as np
import pandas as pd

from simulator import load_csv, fit_unified, unified_predict, ranking_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
DRAWS = 20


def limit_decode(df):
    """Live runs choose rows before their output length is known, so by default every
    fit and test uses the max_tokens limit as decode length (like the reference grids)."""
    return df.assign(decode_tokens=df["max_tokens"].astype(float))


def has_measured(path):
    with open(path) as f:
        return "max_generated_tokens" in f.readline().split(",")


def maxgen_reference(path):
    alt = path.replace(f"{os.sep}results_zeus{os.sep}", f"{os.sep}results_zeus_maxgen{os.sep}")
    return alt if os.path.exists(alt) else path


def score(sample, ref, measured=False):
    if not measured:
        sample, ref = limit_decode(sample), limit_decode(ref)
    seen = set(zip(sample["max_tokens"], sample["batch_size"]))
    test = ref[[s not in seen for s in zip(ref["max_tokens"], ref["batch_size"])]]
    pred = unified_predict(fit_unified(sample, restarts=20), test)
    y = test["measured_energy"].to_numpy(float)
    return {"mape": 100 * np.mean(np.abs(pred - y) / y), **ranking_metrics(test, pred)}


def random_draw(task):
    reference, seed, n, reload_s, cold_s, measured = task
    ref = load_csv(reference)
    sample = ref.iloc[np.random.RandomState(seed).choice(len(ref), n, replace=False)]
    engines = len(sample[["tp_degree", "pp_degree", "budget"]].drop_duplicates())
    minutes = (cold_s + engines * reload_s + sample["measured_latency"].sum()) / 60
    return {"reloads": engines, "minutes": minutes, **score(sample, ref, measured)}


def cold_start_s(rounds):
    t = np.diff([0.0] + [r["elapsed_s"] for r in rounds])
    return max(t[0] - np.median(t[1:]), 0.0) if len(t) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--decode", choices=["limit", "measured"], default="limit")
    args = ap.parse_args()

    live = []
    for path in sorted(glob.glob(os.path.join(HERE, "live_results", "*", "*", "selection_log.json"))):
        log = json.load(open(path))
        s = log["summary"]
        if not s.get("reference"):
            continue
        # Re-score from the measured rows rather than the stored vs_reference, so
        # every run is scored with the current formula, like the random draws.
        sample_csv = os.path.join(os.path.dirname(path), "profiling_metrics.csv")
        sample = load_csv(sample_csv)
        reference = s["reference"]
        measured = False
        if args.decode == "measured":
            reference = maxgen_reference(reference)
            measured = has_measured(reference) and has_measured(sample_csv)
        # Label by folder: the old- and fixed-formula doptimal runs share strategy and seed.
        live.append({"model": s["model"].split("/")[-1], "run": f"{os.path.basename(os.path.dirname(path))} (live)",
                     "reference": reference, "measured": measured, "n": s["valid_rows"], "reloads": s["reloads"],
                     "minutes": s["minutes"], "load_s": s["engine_load_s_median"],
                     "cold_s": cold_start_s(log["rounds"]),
                     **score(sample, load_csv(reference), measured)})
    if not live:
        print("No live runs with a --reference found.")
        return
    live = pd.DataFrame(live)

    rows = []
    for model, runs in live.groupby("model"):
        reload_s = runs["load_s"].median() + 10.0
        measured = bool(runs["measured"].all())
        tasks = [(runs["reference"].iloc[0], seed, int(runs["n"].iloc[0]), reload_s, runs["cold_s"].median(),
                  measured) for seed in range(DRAWS)]
        with mp.Pool(min(DRAWS, len(os.sched_getaffinity(0)))) as pool:
            draws = pd.DataFrame(pool.map(random_draw, tasks))
        rows += runs.drop(columns=["reference", "n", "load_s", "cold_s"]).to_dict("records")
        print(f"  {model}: decode length = {'measured longest output' if measured else 'max_tokens limit'}")
        for stat in ("mean", "median"):
            rows.append({"model": model, "run": f"random (offline, {DRAWS} draws, {stat})", "measured": measured,
                         **getattr(draws, stat)().to_dict()})

    out = pd.DataFrame(rows)[["model", "run", "measured", "reloads", "minutes", "mape", "pairwise", "top1", "regret"]]
    name = "comparison.csv" if args.decode == "limit" else "comparison_measured.csv"
    out.to_csv(os.path.join(HERE, "live_results", name), index=False)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
