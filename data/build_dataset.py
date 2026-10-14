#!/usr/bin/env python3
"""Build the EnergyLens dataset (Parquet + dataset card) from the profiling CSVs.

    python data/build_dataset.py [--out data/hf]
"""
import argparse
import glob
import os
import re
import shutil
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nvidia"))
from validate_results import row_checks  # noqa: E402

MODALITY = {"text_only": "text", "text_image": "image+text", "text_video": "video+text",
            "video_only": "video", "online": "text"}
RUN = re.compile(r"run_(text_only|text_image|text_video|video_only|online)_(.+)_([a-z0-9]+)$")
KEEP = ["hardware", "energy_counter", "model", "modality", "precision", "variant",
        "tensor_parallel_size", "pipeline_parallel_size", "n_devices", "batch_size",
        "max_tokens", "input_length", "generated_tokens", "max_num_batched_tokens", "max_num_seqs",
        "latency", "energy", "batch_energy", "power", "energy_per_output_token", "n_gpus_measured", "power_source",
        "model_params_billion", "active_params_billion", "num_layers", "hidden_size",
        "num_experts", "num_experts_per_tok", "image_resolution", "num_video_frames",
        "frame_width", "frame_height", "actual_image_tokens", "actual_video_tokens", "valid"]


def runs(pattern, hardware, counter):
    for csv in sorted(glob.glob(os.path.join(ROOT, pattern), recursive=True)):
        rel = os.path.relpath(csv, ROOT)
        run = next((p for p in rel.split(os.sep) if RUN.match(p)), None)
        if run is None:
            continue
        mod, model, prec = RUN.match(run).groups()
        sub = os.path.basename(os.path.dirname(csv))
        variant = sub if sub.startswith("res") and sub != "res_native" else ""
        yield csv, dict(hardware=hardware, energy_counter=counter, model=model,
                        modality=MODALITY[mod], precision=prec, variant=variant)


def load_online(csv, meta):
    """Day replay: one row per (window, configuration), energy over the whole window."""
    d = pd.read_csv(csv)
    for k, v in meta.items():
        d[k] = v
    d["n_devices"] = d["tensor_parallel_size"] * d["pipeline_parallel_size"]
    d["energy_per_output_token"] = d["energy"] / d["sum_output"].clip(lower=1)
    return d.drop(columns=["variant"])


def load(csv, meta):
    d = pd.read_csv(csv, low_memory=False)
    d = d.loc[:, ~d.columns.duplicated()]
    if d.empty:
        return d
    if "window_id" in d:
        return load_online(csv, meta)
    for c in ("latency", "power", "energy", "generated_tokens", "max_generated_tokens", "n_gpus_measured"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    flags = row_checks(d)[0]                # error-level checks, one row per input row
    d["valid"] = flags.is_valid.to_numpy()
    for k, v in meta.items():
        d[k] = v
    d["n_devices"] = d["tensor_parallel_size"] * d["pipeline_parallel_size"]
    # `energy` is per request (the batch's energy divided by batch_size), as in the profiler
    d["batch_energy"] = d["energy"] * d["batch_size"]
    d["energy_per_output_token"] = d["batch_energy"] / d["generated_tokens"].clip(lower=1)
    return d[[c for c in KEEP if c in d]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "hf"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sets = {
        "grid": [*runs("nvidia/results_zeus/**/profiling_metrics.csv", "NVIDIA A100-40GB", "Zeus (NVML)"),
                 *runs("intel/results_aurora/**/profiling_metrics.csv", "Intel GPU Max 1550",
                       "Level Zero Sysman")],
        "online_day": [*runs("nvidia/results_zeus_online_day/*/profiling_metrics.csv", "NVIDIA A100-40GB",
                             "Zeus (NVML)")],
    }
    for name, items in sets.items():
        frames = [load(csv, meta) for csv, meta in items]
        df = pd.concat([f for f in frames if len(f)], ignore_index=True)
        path = os.path.join(a.out, f"{name}.parquet")
        df.to_parquet(path, index=False)
        print(f"{name}: {len(df):,} rows ({df.valid.mean() if "valid" in df else 1:.1%} valid), {df.model.nunique()} models -> {path}")
    shutil.copy(os.path.join(ROOT, "data", "DATASET_CARD.md"), os.path.join(a.out, "README.md"))


if __name__ == "__main__":
    main()
