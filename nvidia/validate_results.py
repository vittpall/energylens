#!/usr/bin/env python3
"""Validate profiling datasets before they are used or published.

    python validate_results.py                          # every results_zeus*/ dataset
    python validate_results.py --paths results_zeus/run_text_only_Mistral-7B-Instruct-v0.3_fp16
    python validate_results.py --verbose                # per-check counts for every dataset
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TDP_W = {"A100": 400.0,           # A100-SXM4-40GB
         "Max1550": 600.0}        # Intel Data Center GPU Max 1550, per card (two tiles)
# Aurora: vLLM puts one rank per tile and the Level Zero counter reads whole cards
TILES_PER_CARD = {"A100": 1, "Max1550": 2}
SOURCES = {"A100": ("zeus_sampling",), "Max1550": ("ze_counter_poll", "ze_counter_card")}
MIN_W = 40.0                     # below an A100's idle draw: a measurement error
MISMATCH = 0.25
OUTLIER_Z = 4.0
SHORT_S = 0.5
LAYOUT_COVERAGE = 0.8            # a layout with < 80% of the best layout's rows is flagged


def load(path):
    d = pd.read_csv(path)
    d = d.loc[:, ~d.columns.duplicated()]
    for c in ("latency", "power", "energy", "generated_tokens", "max_generated_tokens", "n_gpus_measured"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def row_checks(d, gpu=None):
    if gpu is None:     # Intel runs record a Level Zero power source
        intel = "power_source" in d and d.power_source.astype(str).str.startswith("ze_counter").any()
        gpu = "Max1550" if intel else "A100"
    n = len(d)
    err, warn = {}, {}
    missing = d[["latency", "power", "energy"]].isna().any(axis=1)
    err["missing_measurement"] = missing
    ok = ~missing
    err["zero_energy"] = ok & (d.energy <= 0)
    g = np.ceil(d.tensor_parallel_size * d.pipeline_parallel_size / TILES_PER_CARD[gpu])
    if "n_gpus_measured" in d:
        err["gpu_count"] = ok & (d.n_gpus_measured != g)
    if "power_source" in d:
        err["power_source"] = ok & ~d.power_source.astype(str).isin(SOURCES[gpu])
    per_gpu = d.power / g
    err["power_range"] = ok & ((per_gpu < MIN_W) | (per_gpu > TDP_W[gpu] * 1.1))
    ratio = d.energy * d.batch_size / (d.power * d.latency)
    err["energy_power_mismatch"] = ok & (d.energy > 0) & ((ratio - 1).abs() > MISMATCH)
    tok = pd.Series(False, index=d.index)
    if "generated_tokens" in d:
        tok |= d.generated_tokens > d.batch_size * d.max_tokens
    if "max_generated_tokens" in d:
        tok |= d.max_generated_tokens > d.max_tokens
    err["token_count"] = ok & tok

    good = ok & (d.energy > 0)
    keys = ["tensor_parallel_size", "pipeline_parallel_size", "max_num_batched_tokens", "max_tokens", "batch_size"]
    le = np.log(d.energy.where(good))
    med = le.groupby([d[k] for k in keys]).transform("median")
    mad = (le - med).abs().groupby([d[k] for k in keys]).transform("median")
    cnt = le.groupby([d[k] for k in keys]).transform("count")
    z = 0.6745 * (le - med) / mad.replace(0, np.nan)
    warn["energy_outlier"] = good & (cnt >= 4) & (z.abs() > OUTLIER_Z)
    warn["short_window"] = good & (d.latency < SHORT_S)
    if "max_generated_tokens" in d:
        warn["early_stop"] = good & (d.max_generated_tokens < d.max_tokens)
    if "max_num_seqs" in d:
        warn["waves"] = good & (d.batch_size > d.max_num_seqs)

    errors = pd.DataFrame(err).fillna(False).astype(bool)
    warns = pd.DataFrame(warn).fillna(False).astype(bool)
    flags = pd.DataFrame({"row": np.arange(n), "is_valid": ~errors.any(axis=1).to_numpy()})
    flags["errors"] = errors.apply(lambda r: ";".join(errors.columns[r.to_numpy()]), axis=1).to_numpy()
    flags["warnings"] = warns.apply(lambda r: ";".join(warns.columns[r.to_numpy()]), axis=1).to_numpy()
    return flags, errors.sum().to_dict(), warns.sum().to_dict(), ratio[good]


def dataset_checks(d, flags, run_dir):
    issues = []
    v = d[flags.is_valid.to_numpy()]
    lay = v.groupby(["tensor_parallel_size", "pipeline_parallel_size"]).size()
    if len(lay):
        low = lay[lay < LAYOUT_COVERAGE * lay.max()]
        for (t, p), c in low.items():
            issues.append(f"layout ({t},{p}): {c} valid rows vs {lay.max()} for the best-covered layout")
        cov = v.pivot_table(index=["tensor_parallel_size", "pipeline_parallel_size"],
                            columns="max_num_batched_tokens", values="energy", aggfunc="size").fillna(0)
        for (t, p), r in cov.iterrows():
            for b, c in r.items():
                ref = cov[b].max()
                if c < LAYOUT_COVERAGE * ref:
                    issues.append(f"layout ({t},{p}) budget {b}: {int(c)}/{int(ref)} rows")
    sc = v.drop_duplicates(["max_tokens", "batch_size", "tensor_parallel_size", "pipeline_parallel_size"]) \
          .groupby(["max_tokens", "batch_size"]).size()
    single = int((sc < 2).sum())
    dup_cols = [c for c in d.columns if c not in ("latency", "power", "energy", "generated_tokens",
                                                  "max_generated_tokens")]
    dups = int(d.duplicated(subset=dup_cols).sum())
    prog = {}
    pf = os.path.join(run_dir, "profiling_progress.json")
    if os.path.exists(pf):
        p = json.load(open(pf))
        failed = set(p.get("failed_configs", []))
        skipped = sum(len(x) for x in p.get("skipped_batches", {}).values())
        prog = {"configs_total": p.get("total_configs"), "configs_failed": len(failed),
                "batches_skipped": skipped}
    return issues, {"scenarios": len(sc), "single_layout_scenarios": single, "duplicate_rows": dups, **prog}


def validate(path, verbose=False, out_dir=None):
    d = load(path)
    run_dir = os.path.dirname(path)
    flags, errs, warns, ratio = row_checks(d)
    issues, stats = dataset_checks(d, flags, run_dir)
    name = os.path.relpath(run_dir, HERE).replace(os.sep, "__")
    if out_dir:
        flags.to_csv(os.path.join(out_dir, f"{name}.flags.csv"), index=False)
    summary = {"dataset": os.path.relpath(run_dir, HERE), "rows": len(d), "valid_rows": int(flags.is_valid.sum()),
               "valid_pct": round(100 * flags.is_valid.mean(), 1),
               **{f"err_{k}": int(v) for k, v in errs.items()},
               **{f"warn_{k}": int(v) for k, v in warns.items()},
               "energy_power_ratio_median": round(float(ratio.median()), 3) if len(ratio) else np.nan,
               **stats, "coverage_issues": len(issues)}
    status = "PASS" if summary["valid_pct"] >= 99 and not issues else ("WARN" if summary["valid_pct"] >= 90 else "FAIL")
    summary["status"] = status
    bad = {k: v for k, v in errs.items() if v}
    print(f"[{status}] {summary['dataset']}: {summary['valid_rows']}/{summary['rows']} valid"
          + (f"; errors {bad}" if bad else "")
          + (f"; {len(issues)} coverage issue(s)" if issues else ""))
    if verbose or status != "PASS":
        w = {k: v for k, v in warns.items() if v and k != "waves"}
        if w:
            print(f"        warnings: {w}")
        for i in issues[:8]:
            print(f"        - {i}")
        if len(issues) > 8:
            print(f"        ... {len(issues) - 8} more")
        if stats.get("single_layout_scenarios"):
            print(f"        {stats['single_layout_scenarios']}/{stats['scenarios']} scenarios have one layout")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--paths", nargs="+", default=None,
                    help="profiling_metrics.csv files or run folders (default: every results_zeus*/ run)")
    ap.add_argument("--out", default=os.path.join(HERE, "results_validation"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.paths:
        paths = []
        for p in args.paths:
            paths += [p] if p.endswith(".csv") else glob.glob(os.path.join(p, "**", "profiling_metrics.csv"),
                                                               recursive=True)
    else:
        paths = sorted(p for p in glob.glob(os.path.join(HERE, "results_zeus*", "run_*", "*", "profiling_metrics.csv")))
    os.makedirs(args.out, exist_ok=True)
    rows = [validate(p, args.verbose, args.out) for p in paths]
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print(f"\n{(s.status == 'PASS').sum()} PASS, {(s.status == 'WARN').sum()} WARN, {(s.status == 'FAIL').sum()} FAIL "
          f"of {len(s)} datasets; {s.valid_rows.sum()}/{s.rows.sum()} rows valid  -> {args.out}/summary.csv")


if __name__ == "__main__":
    main()
