#!/usr/bin/env python3
"""
Cross-model baseline comparison: EnergyLens vs ML baselines vs MaverIQ
========================================================================
Reproduces the paper tables:
  - Table 1: MAPE / R² / RMSE at n_train=50 and n_train=500 per dataset
  - Table 2: MAPE / R² at n_train=50 across all hardware-model combinations

Per seed, a fixed test set (--test-frac of the rows, default 0.2) is held out;
every method is fit on the same n_train rows drawn from the remaining pool and
scored only on the held-out rows. --test-frac 0 restores the old behaviour
(score on the full dataset, training rows included).
"""

DATASETS = {
    # ── NVIDIA A100 (results/) ──
    "Mistral-7B (NVIDIA A100)":    "./results/run_text_only_Mistral-7B-Instruct-v0.3/profiling_metrics.csv",
    "Nemotron-Nano-9B (NVIDIA)":   "./results/run_text_only_NVIDIA-Nemotron-Nano-9B-v2/profiling_metrics.csv",
    "Mamba-130M (NVIDIA)":         "./results/run_text_only_mamba-130m-hf/profiling_metrics.csv",
    "Falcon-Mamba-7B":             "./results/run_text_only_falcon-mamba-7b/res_native/profiling_metrics.csv",
    "Qwen1.5-MoE-2.7B (NVIDIA)":   "./results/run_text_only_Qwen1.5-MoE-A2.7B/profiling_metrics.csv",
    "Mixtral-8x7B-Instruct-v0.1":  "./results/run_text_only_Mixtral-8x7B-Instruct-v0.1/res_native/profiling_metrics.csv",
    "Qwen2.5-14B-text":            "./results/run_text_only_Qwen2.5-14B-Instruct/res_native/profiling_metrics.csv",
    "Qwen2.5-32B-text":            "./results/run_text_only_Qwen2.5-32B-Instruct/res_native/profiling_metrics.csv",
    "Qwen2-VL-7B text (NVIDIA)":   "./results/run_text_only_Qwen2-VL-7B-Instruct/profiling_metrics.csv",
    "Qwen2-VL-7B image (NVIDIA)":  "./results/run_text_image_Qwen2-VL-7B-Instruct/profiling_metrics.csv",
    "Qwen2-VL-7B video (NVIDIA)":  "./results/run_text_video_Qwen2-VL-7B-Instruct/combined_metrics.csv",
    "Qwen2-VL-7B video-only (NVIDIA)": "./results/run_video_only_Qwen2-VL-7B-Instruct/profiling_metrics.csv",
    "LLaVA-1.5-7B text (NVIDIA)":  "./results/run_text_only_llava-1.5-7b-hf/profiling_metrics.csv",
    "LLaVA-1.5-7B image (NVIDIA)": "./results/run_text_image_llava-1.5-7b-hf/profiling_metrics.csv",
    "LLaVA-1.5-7B video (NVIDIA)": "./results/run_text_video_llava-1.5-7b-hf/combined_metrics.csv",
    "InternVL2-Llama3-76B image":  "./results/run_text_image_InternVL2-Llama3-76B/res_native/profiling_metrics.csv",
    # ── Quantized (bnb 4-bit) ──
    "Mistral-7B-bnb4bit":          "./results/run_text_only_Mistral-7B-Instruct-v0.3_fp16-bnb4bit/res_native/profiling_metrics.csv",
    "Qwen2.5-14B-bnb4bit":         "./results/run_text_only_Qwen2.5-14B-Instruct_fp16-bnb4bit/res_native/profiling_metrics.csv",
    # ── Other hardware ──
    "Mistral-7B-image-chat (AMD MI100)":  "./results/amd/profiling_metrics.csv",
    "Mistral-7B-video-chat (INTEL 1550)": "./results/intel/profiling_metrics.csv",
    # ── Zeus-measured energy (results_zeus/) ──
    "Mistral-7B-fp16 (zeus)":       "./results_zeus/run_text_only_Mistral-7B-Instruct-v0.3_fp16/res_native/profiling_metrics.csv",
    "Mistral-Small-24B-fp16 (zeus)":"./results_zeus/run_text_only_Mistral-Small-24B-Instruct-2501_fp16/res_native/profiling_metrics.csv",
    "Nemotron-Nano-9B-fp16 (zeus)": "./results_zeus/run_text_only_NVIDIA-Nemotron-Nano-9B-v2_fp16/res_native/profiling_metrics.csv",
    "Qwen2.5-14B-fp16 (zeus)":      "./results_zeus/run_text_only_Qwen2.5-14B-Instruct_fp16/res_native/profiling_metrics.csv",
}

# Full learning curve: dense sampling below, at, and well beyond the point
# where the ML baselines (Gradient Boosting in particular) catch up. Stopping
# at 500 — where the advantage disappears — is not enough to support the
# small-sample claim; the curve must show the whole trajectory.
N_TRAIN_VALS = [25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700]
N_SEEDS      = 5   # average over multiple random seeds; also gives error bands
TEST_FRAC    = 0.2 # rows held out per seed; every method is scored only on these

import argparse, os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize as opt
warnings.filterwarnings('ignore')

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    _SKLEARN = True
except ImportError:
    _SKLEARN = False
    print("[WARNING] scikit-learn not found – ML baselines will be skipped.")


# ════════════════════════════════════════════════════════════
#  METRICS
# ════════════════════════════════════════════════════════════

def compute_metrics(pred, true):
    mask = true > 0
    mape = np.mean(np.abs(pred[mask] - true[mask]) / true[mask]) * 100
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2   = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    return mape, r2, rmse


# ════════════════════════════════════════════════════════════
#  COLUMN MAPPING
# ════════════════════════════════════════════════════════════

# vLLM runs at most max_num_seqs requests at once. source/model.py forces 64 on
# the NVIDIA profiler, but the CSVs record 0 in that column. A batch above the
# cap runs in waves, so each request gets the batching gain of a cap-sized batch,
# not of the whole batch: the formula uses min(batch_size, max_num_seqs).
PROFILER_MAX_NUM_SEQS = 64
# The AMD profiler keeps vLLM's default (256), above its largest batch (207).
UNCAPPED_SOURCES = ("/amd/",)


def max_num_seqs_column(df, data_path):
    """Per-row concurrency cap: the recorded value when set, else the profiler's."""
    recorded = (pd.to_numeric(df['max_num_seqs'], errors='coerce').fillna(0).values
                if 'max_num_seqs' in df.columns else np.zeros(len(df)))
    default = (np.inf if any(s in data_path for s in UNCAPPED_SOURCES)
               else PROFILER_MAX_NUM_SEQS)
    return np.where(recorded > 0, recorded, default)


def effective_batch(df):
    """Requests that actually share the GPU: batch_size capped at max_num_seqs."""
    bs = df['batch_size'].values.astype(float)
    cap = (df['max_num_seqs'].values.astype(float) if 'max_num_seqs' in df.columns
           else PROFILER_MAX_NUM_SEQS)
    return np.minimum(bs, cap)


# Vision tokens. The NVIDIA profiler's input_length is the batch's total input
# and already contains every request's image/video tokens; the AMD and Intel
# profilers record text tokens only. *_token_count_estimate is a model-level
# constant written on every row (576 even for text-only models), not a
# measurement, so it is never used.
TEXT_ONLY_INPUT_SOURCES = ("/amd/", "/intel/")


def vision_tokens(df):
    """Per-request (image, video) tokens from the measured columns."""
    def pick(cols):
        col = next((c for c in cols if c in df.columns), None)
        return (pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float).values
                if col else np.zeros(len(df)))
    return (pick(['actual_image_tokens', 'image_token_count']),
            pick(['actual_video_tokens', 'video_token_count']))


def total_input_tokens(input_length, vis_per_request, batch_size, data_path):
    """Batch input tokens with each request's vision tokens counted once."""
    input_length = np.asarray(input_length, dtype=float)
    if any(s in data_path for s in TEXT_ONLY_INPUT_SOURCES):
        return input_length + np.asarray(vis_per_request, float) * np.asarray(batch_size, float)
    return input_length


def decode_tokens(df, max_tokens):
    """Decode steps per batch: the longest output if the profiler recorded it,
    else the max_tokens limit (older runs, where early stops go unnoticed)."""
    limit = np.asarray(max_tokens, dtype=float)
    if 'max_generated_tokens' not in df.columns:
        return limit
    gen = pd.to_numeric(df['max_generated_tokens'], errors='coerce').values.astype(float)
    return np.maximum(np.where(np.isnan(gen), limit, gen), 1.0)


def drop_invalid_rows(df, data_path):
    """Drop the rows validate_results.py marks invalid (its error-level row checks).

    Applied only to the current profiler's output (it records power_source); older
    or external data passes through. Set ENERGYLENS_KEEP_INVALID=1 to keep every row
    (e.g. to reproduce numbers computed before this filter existed)."""
    if 'power_source' not in df.columns or os.environ.get('ENERGYLENS_KEEP_INVALID') == '1':
        return df
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from validate_results import row_checks
    d = df.loc[:, ~df.columns.duplicated()].copy()
    for c in ('latency', 'power', 'energy', 'generated_tokens', 'max_generated_tokens', 'n_gpus_measured'):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    flags = row_checks(d)[0]
    # missing/zero measurements are dropped (and reported) by prepare_df itself
    bad = ~flags.is_valid.to_numpy() & ~d[['latency', 'power', 'energy']].isna().any(axis=1).to_numpy() \
          & (d['energy'].to_numpy() > 0)
    if bad.any():
        reasons = flags.errors[bad].str.split(';').explode().value_counts().to_dict()
        print(f"    [prepare_df] dropped {bad.sum()}/{len(df)} rows invalid per validate_results "
              f"{reasons} in {os.path.basename(os.path.dirname(data_path)) or data_path}")
    return df[~bad].reset_index(drop=True)


def prepare_df(df, data_path):
    out = pd.DataFrame()
    df = drop_invalid_rows(df, data_path)
    required = {'latency': 'measured_latency', 'energy': 'measured_energy',
                'power': 'gpu_power_w', 'tensor_parallel_size': 'tp_degree',
                'pipeline_parallel_size': 'pp_degree', 'batch_size': 'batch_size'}
    for src, dst in required.items():
        if src not in df.columns:
            raise ValueError(f"Missing column '{src}' in {data_path}")
        dtype = int if dst in ['tp_degree', 'pp_degree', 'batch_size'] else float
        out[dst] = df[src].astype(dtype)

    for alt in ['max_tokens', 'output_length', 'output_tokens']:
        if alt in df.columns:
            out['max_tokens'] = df[alt].astype(int); break
    else:
        out['max_tokens'] = 1

    for alt in ['input_length', 'total_input_tokens']:
        if alt in df.columns:
            out['input_length'] = df[alt].astype(float); break
    else:
        out['input_length'] = 0.0

    out['image_tokens'], out['video_tokens'] = vision_tokens(df)
    out['vis_tokens']   = np.maximum(out['image_tokens'], out['video_tokens'])
    out['total_input']  = total_input_tokens(out['input_length'], out['vis_tokens'],
                                             out['batch_size'], data_path)
    out['max_num_seqs'] = max_num_seqs_column(df, data_path)
    out['decode_tokens'] = decode_tokens(df, out['max_tokens'])

    # Failed-generation rows have blank latency/power/energy (the profiler writes
    # empty cells on a generation exception), which become NaN here. They carry
    # no measurement: sklearn refuses any NaN in the target and NaN propagates
    # into RMSE for the formula models. Drop them so every method sees clean data.
    critical = ['measured_latency', 'gpu_power_w', 'measured_energy']
    # A window shorter than the GPU energy counter's update interval reads 0 J
    # (seen on ~50 ms single-request batches); that is a missing measurement too.
    out.loc[out['measured_energy'] <= 0, 'measured_energy'] = np.nan
    before = len(out)
    out = out.dropna(subset=critical).reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        print(f"    [prepare_df] dropped {dropped}/{before} rows with missing "
              f"latency/power/energy in {os.path.basename(data_path)}")

    return out


# ════════════════════════════════════════════════════════════
#  MODEL: EnergyLens — Prefill/Decode Split (12 params)
# ════════════════════════════════════════════════════════════

def unified_predict(p, df):
    α_p, β_p, γ1_p, γ2_p, ε_p, α_d, β_d, γ1_d, γ2_d, ε_d, δ1, δ2 = p
    bs  = effective_batch(df)
    tok = df['decode_tokens'].values.astype(float)
    tp  = df['tp_degree'].values.astype(float)
    pp  = df['pp_degree'].values.astype(float)
    inp = df['total_input'].values.astype(float)
    E_pf = α_p * inp / (np.power(bs, β_p) + ε_p) * np.power(tp, γ1_p) * np.power(pp, γ2_p)
    E_dc = α_d * tok / (np.power(bs, β_d) + ε_d) * np.power(tp, γ1_d) * np.power(pp, γ2_d)
    return E_pf + E_dc + δ1 * tp + δ2 * pp


def fit_unified(df, restarts=20):
    true = df['measured_energy'].values.astype(float)
    bs   = effective_batch(df)
    tok  = df['decode_tokens'].values.astype(float)
    tp   = df['tp_degree'].values.astype(float)
    pp   = df['pp_degree'].values.astype(float)
    inp  = df['total_input'].values.astype(float)
    mask = true > 0
    med     = np.median(true[mask]) if mask.any() else 1.0
    med_tok = np.median(tok[mask])  if mask.any() else 1.0
    med_inp = np.median(inp[mask])  if mask.any() else 1.0

    def loss(p):
        α_p, β_p, γ1_p, γ2_p, ε_p, α_d, β_d, γ1_d, γ2_d, ε_d, δ1, δ2 = p
        E_pf = α_p * inp / (np.power(bs, β_p) + ε_p) * np.power(tp, γ1_p) * np.power(pp, γ2_p)
        E_dc = α_d * tok / (np.power(bs, β_d) + ε_d) * np.power(tp, γ1_d) * np.power(pp, γ2_d)
        pred = E_pf + E_dc + δ1 * tp + δ2 * pp
        return np.mean(np.abs(pred[mask] - true[mask]) / true[mask])

    scale_p = med * 0.3 / max(med_inp, 1)
    scale_d = med * 0.7 / max(med_tok, 1)
    bounds  = [
        (1e-8, 1e6), (0.01, 3), (-2, 3), (-2, 3), (0, 100),
        (1e-8, 1e6), (0.01, 3), (-2, 3), (-2, 3), (0, 100),
        (-1e6, 1e6), (-1e6, 1e6),
    ]
    best, best_p = np.inf, None
    for i in range(restarts):
        rng = np.random.RandomState(i * 42)
        x0 = [
            scale_p * rng.uniform(0.1, 3.0), rng.uniform(0.3, 1.5),
            rng.uniform(-1.2, -0.2), rng.uniform(-1.0, 0.0), rng.uniform(0.01, 2.0),
            scale_d * rng.uniform(0.1, 3.0), rng.uniform(0.3, 1.5),
            rng.uniform(-1.2, -0.2), rng.uniform(-1.0, 0.0), rng.uniform(0.01, 2.0),
            rng.uniform(-med * 0.05, med * 0.1), rng.uniform(-med * 0.05, med * 0.1),
        ]
        r = opt(loss, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 10000, 'ftol': 1e-15})
        if r.fun < best:
            best, best_p = r.fun, r.x
    return best_p


# ════════════════════════════════════════════════════════════
#  MODEL: MaverIQ — single-stage direct energy fit (12 params)
# ════════════════════════════════════════════════════════════

def maveriq_predict(p, df):
    (L1_base, L1_tp_ov, L1_pp_ov,
     L2_base, L2_tp_ov, L2_pp_ov,
     L_bs, alpha, beta, gamma, delta, eta) = p
    tp      = df['tp_degree'].values.astype(float)
    pp      = df['pp_degree'].values.astype(float)
    bs      = df['batch_size'].values.astype(float)
    tok     = df['decode_tokens'].values.astype(float)
    pp_safe = np.maximum(pp, 1.0)
    tp_m1   = np.maximum(tp - 1.0, 0.0)
    pp_m1   = np.maximum(pp - 1.0, 0.0)
    L1 = (L1_base / (tp * np.power(pp_safe, alpha))
          + np.where(tp > 1, np.power(tp_m1, beta) / np.power(pp_safe, gamma) * L1_tp_ov, 0.0)
          + np.where(pp > 1, np.power(pp_m1, delta) * L1_pp_ov, 0.0))
    L2 = (L2_base / (tp * np.power(pp_safe, alpha))
          + np.where(tp > 1, np.power(tp_m1, beta) / np.power(pp_safe, gamma) * L2_tp_ov, 0.0)
          + np.where(pp > 1, np.power(pp_m1, delta) * L2_pp_ov, 0.0))
    lat = L1 + tok * L2 + np.maximum(bs - 1.0, 0.0) * L_bs
    return np.maximum(eta * lat, 1e-9)


def fit_maveriq(df, restarts=5):
    true    = df['measured_energy'].values.astype(float)
    lat_arr = df['measured_latency'].values.astype(float)
    mask    = true > 0
    med_e   = np.median(true[mask]) if mask.any() else 1.0
    med_l   = np.median(lat_arr[lat_arr > 0]) if (lat_arr > 0).any() else 1.0
    med_tok = max(np.median(df['decode_tokens'].values), 1)
    eta0    = med_e / max(med_l, 1e-6)
    bounds  = [
        (1e-6, 1e4), (0.0, 1e3), (0.0, 1e3),
        (1e-6, 1e4), (0.0, 1e3), (0.0, 1e3),
        (0.0, 1e3), (0.01, 2.0), (0.01, 3.0),
        (0.01, 2.0), (0.01, 2.0), (1e-3, 1e6),
    ]

    def loss(p):
        pred = maveriq_predict(p, df)
        return np.mean(np.abs(pred[mask] - true[mask]) / true[mask])

    best, best_p = np.inf, None
    for i in range(restarts):
        rng = np.random.RandomState(i * 17)
        L1_0 = med_l * rng.uniform(0.1, 0.5)
        L2_0 = med_l * rng.uniform(0.1, 0.8) / med_tok
        x0 = [
            L1_0, L1_0 * rng.uniform(0.01, 0.2), L1_0 * rng.uniform(0.01, 0.2),
            L2_0, L2_0 * rng.uniform(0.01, 0.2), L2_0 * rng.uniform(0.01, 0.2),
            med_l * rng.uniform(0.001, 0.05),
            rng.uniform(0.3, 1.2), rng.uniform(0.5, 1.5),
            rng.uniform(0.3, 1.0), rng.uniform(0.3, 1.0),
            eta0 * rng.uniform(0.5, 2.0),
        ]
        r = opt(loss, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 8000, 'ftol': 1e-13})
        if r.fun < best:
            best, best_p = r.fun, r.x
    return best_p


# ════════════════════════════════════════════════════════════
#  ML BASELINES
# ════════════════════════════════════════════════════════════

def get_ml_features(df):
    return np.column_stack([
        df['batch_size'].values.astype(float),
        df['decode_tokens'].values.astype(float),
        df['tp_degree'].values.astype(float),
        df['pp_degree'].values.astype(float),
        df['total_input'].values.astype(float),
        df['vis_tokens'].values.astype(float),
    ])


def fit_predict_ml(train_df, full_df, true_all, train_mape=None, fit_time=None):
    """Held-out metrics per ML baseline; fills `train_mape` (if given) with the MAPE on the
    training rows and `fit_time` (if given) with the fitting wall-clock seconds."""
    results = {}
    if not _SKLEARN:
        return results
    X_train = get_ml_features(train_df)
    X_all   = get_ml_features(full_df)
    y_train = train_df['measured_energy'].values.astype(float)
    sc = StandardScaler().fit(X_train)
    Xtr = sc.transform(X_train)
    Xall = sc.transform(X_all)
    for name, cls in [
        ('Linear Regression',     LinearRegression()),
        ('Random Forest',         RandomForestRegressor(n_estimators=100, random_state=42)),
        ('Gradient Boosting',     GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ]:
        try:
            t0 = time.perf_counter()
            cls.fit(Xtr, y_train)
            if fit_time is not None:
                fit_time[name] = time.perf_counter() - t0
            pred = cls.predict(Xall)
            results[name] = compute_metrics(pred, true_all)
            if train_mape is not None:
                train_mape[name] = compute_metrics(cls.predict(Xtr), y_train)[0]
        except Exception as e:
            print(f"    [WARN] {name} failed: {e}")
            results[name] = (np.nan, np.nan, np.nan)
    return results


# ════════════════════════════════════════════════════════════
#  TRAIN / TEST SPLITS (shared with the other test scripts)
# ════════════════════════════════════════════════════════════

def holdout_rows(df, rng, test_frac=TEST_FRAC):
    """
    Random row holdout. Returns (pool_df, test_df): training rows must be drawn
    from pool_df only, and every method is scored on test_df only.

    The permutation is drawn from `rng` first, so for a given seed the test set
    is the same whatever is drawn afterwards (any n_train, any sampling
    strategy). test_frac <= 0 returns (df, df): the old in-sample behaviour.
    """
    if not test_frac or test_frac <= 0:
        return df, df
    n      = len(df)
    perm   = rng.permutation(n)
    n_test = max(1, int(round(test_frac * n)))
    return (df.iloc[perm[n_test:]].reset_index(drop=True),
            df.iloc[perm[:n_test]].reset_index(drop=True))


def holdout_scenarios(df, test_frac=0.3, seed=42):
    """
    Scenario holdout for the ranking scripts. A scenario is one
    (max_tokens, batch_size) workload point; all its rows (every TP/PP config)
    go to the same side, so held-out scenarios keep their full config set to
    rank. A row-level split would not work here: there is ~1 row per
    (scenario, config) cell, so it would strip configs out of scenarios.

    Returns (train_df, test_df, n_test, n_scen). Falls back to in-sample
    (df, df, 0, n_scen) when test_frac <= 0 or there are fewer than 4
    scenarios to split.
    """
    keys      = list(zip(df['max_tokens'], df['batch_size']))
    scenarios = sorted(set(keys))
    n_scen    = len(scenarios)
    if not test_frac or test_frac <= 0 or n_scen < 4:
        return df, df, 0, n_scen
    perm   = np.random.RandomState(seed).permutation(n_scen)
    n_test = min(max(1, int(round(test_frac * n_scen))), n_scen - 2)
    test_keys = set(scenarios[i] for i in perm[:n_test])
    mask = np.array([k in test_keys for k in keys])
    return (df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True),
            n_test, n_scen)


# ════════════════════════════════════════════════════════════
#  RUN ONE DATASET AT ONE n_train (averaged over seeds)
# ════════════════════════════════════════════════════════════

def evaluate_at_n(df, n_train, seeds=N_SEEDS, test_frac=TEST_FRAC):
    """
    Fit every method on the same n_train rows and score it on held-out rows.

    Per seed, a fixed test set (test_frac of the rows) is drawn first and the
    n_train training rows come only from the remaining pool, so no method is
    ever scored on rows it was fit on. The seed's rng draws the test set first,
    so for a given seed the test set is identical at every n_train and the
    learning curve compares like with like.

    test_frac=0 reproduces the old behaviour (score on the full df, which
    includes the training rows). Returns None when n_train exceeds the pool.
    Also returns the MAPE on the training rows, so the train-vs-held-out gap
    (overfitting) can be reported next to the held-out error, and each method's
    mean fitting wall-clock time in seconds.
    """
    accum, accum_train, accum_time = {}, {}, {}

    for seed in range(seeds):
        rng = np.random.RandomState(seed * 7 + 13)
        pool_df, test_df = holdout_rows(df, rng, test_frac)
        if test_frac and test_frac > 0 and n_train > len(pool_df):
            return None
        if n_train >= len(pool_df):
            train_df = pool_df
        else:
            idx      = rng.choice(len(pool_df), size=n_train, replace=False)
            train_df = pool_df.iloc[idx].reset_index(drop=True)
        true_e  = test_df['measured_energy'].values.astype(float)
        true_tr = train_df['measured_energy'].values.astype(float)

        # EnergyLens
        try:
            t0   = time.perf_counter()
            p    = fit_unified(train_df)
            accum_time.setdefault('EnergyLens (Ours)', []).append(time.perf_counter() - t0)
            pred = unified_predict(p, test_df)
            m    = compute_metrics(pred, true_e)
            tr   = compute_metrics(unified_predict(p, train_df), true_tr)[0]
        except Exception as e:
            print(f"    [WARN] EnergyLens failed (seed {seed}): {e}")
            m, tr = (np.nan, np.nan, np.nan), np.nan
        accum.setdefault('EnergyLens (Ours)', []).append(m)
        accum_train.setdefault('EnergyLens (Ours)', []).append(tr)

        # MaverIQ
        try:
            t0   = time.perf_counter()
            p    = fit_maveriq(train_df)
            accum_time.setdefault('MaverIQ (Lat×P)', []).append(time.perf_counter() - t0)
            pred = maveriq_predict(p, test_df)
            m    = compute_metrics(pred, true_e)
            tr   = compute_metrics(maveriq_predict(p, train_df), true_tr)[0]
        except Exception as e:
            print(f"    [WARN] MaverIQ failed (seed {seed}): {e}")
            m, tr = (np.nan, np.nan, np.nan), np.nan
        accum.setdefault('MaverIQ (Lat×P)', []).append(m)
        accum_train.setdefault('MaverIQ (Lat×P)', []).append(tr)

        # ML baselines
        ml_train, ml_time = {}, {}
        ml = fit_predict_ml(train_df, test_df, true_e, train_mape=ml_train, fit_time=ml_time)
        for name, m in ml.items():
            accum.setdefault(name, []).append(m)
            accum_train.setdefault(name, []).append(ml_train.get(name, np.nan))
            accum_time.setdefault(name, []).append(ml_time.get(name, np.nan))

    # Average metrics across seeds; also keep the MAPE std for error bands.
    averaged = {}
    mape_std = {}
    for name, runs in accum.items():
        arr = np.array(runs, dtype=float)
        averaged[name] = tuple(np.nanmean(arr, axis=0))
        mape_std[name] = float(np.nanstd(arr[:, 0]))   # std of MAPE across seeds
    train_mape = {name: float(np.nanmean(v)) for name, v in accum_train.items()}
    fit_time   = {name: float(np.nanmean(v)) for name, v in accum_time.items()}
    return averaged, mape_std, train_mape, fit_time


# ════════════════════════════════════════════════════════════
#  PLOTTING
# ════════════════════════════════════════════════════════════

COLORS = {
    'EnergyLens (Ours)': '#2a9d8f',
    'Linear Regression':  '#457b9d',
    'Random Forest':      '#e9c46a',
    'Gradient Boosting':  '#f4a261',
    'MaverIQ (Lat×P)':   '#e76f51',
}
MARKERS = {
    'EnergyLens (Ours)': 'o',
    'Linear Regression':  's',
    'Random Forest':      '^',
    'Gradient Boosting':  'D',
    'MaverIQ (Lat×P)':   'x',
}


def plot_mape_bar(results_by_dataset, n_train, output_dir):
    """Bar chart: MAPE of each method per dataset at a fixed n_train."""
    plt.ioff()
    datasets = list(results_by_dataset.keys())
    methods  = list(next(iter(results_by_dataset.values())).keys())
    x   = np.arange(len(datasets))
    w   = 0.15
    n_m = len(methods)
    offsets = np.linspace(-(n_m - 1) / 2 * w, (n_m - 1) / 2 * w, n_m)

    fig, ax = plt.subplots(figsize=(max(12, len(datasets) * 2), 6))
    for i, method in enumerate(methods):
        mapes = [results_by_dataset[ds][method][0] for ds in datasets]
        ax.bar(x + offsets[i], mapes, w, label=method,
               color=COLORS.get(method, f'C{i}'), edgecolor='black', lw=0.5)

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('MAPE (%)', fontsize=12)
    ax.set_title(f'Cross-Model Comparison at n_train={n_train}\n(lower MAPE is better)',
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=9)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', ls='--', alpha=0.4)
    fig.tight_layout()
    out = os.path.join(output_dir, f'mape_bar_n{n_train}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_mape_heatmap(all_results, output_dir):
    """
    Heatmap: rows = datasets, cols = (method, n_train) pairs.
    Cell value = MAPE.
    """
    plt.ioff()
    datasets = list(all_results.keys())
    n_trains = N_TRAIN_VALS
    methods  = list(next(iter(next(iter(all_results.values())).values())).keys())

    cols  = [(m, n) for n in n_trains for m in methods]
    data  = np.full((len(datasets), len(cols)), np.nan)
    for ri, ds in enumerate(datasets):
        for ci, (m, n) in enumerate(cols):
            if n in all_results[ds] and m in all_results[ds][n]:
                data[ri, ci] = all_results[ds][n][m][0]   # MAPE

    col_labels = [f"{m}\nn={n}" for m, n in cols]
    fig, ax = plt.subplots(figsize=(max(16, len(cols) * 1.4), max(5, len(datasets) * 0.8)))
    vmin, vmax = np.nanmin(data), min(np.nanmax(data), 200)
    im = ax.imshow(data, cmap='RdYlGn_r', vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax, label='MAPE (%)')
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha='right')
    ax.set_yticks(range(len(datasets))); ax.set_yticklabels(datasets, fontsize=9)
    for ri in range(len(datasets)):
        for ci in range(len(cols)):
            v = data[ri, ci]
            if not np.isnan(v):
                tc = 'white' if (v - vmin) / max(vmax - vmin, 1e-9) > 0.65 else 'black'
                ax.text(ci, ri, f'{v:.1f}', ha='center', va='center', fontsize=6, color=tc)
    ax.set_title('MAPE Heatmap: All Methods × All Datasets × n_train', fontsize=12)
    fig.tight_layout()
    out = os.path.join(output_dir, 'mape_heatmap_all.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out}")


# Learning-curve style — same fonts, line/marker style, grid and palette as
# viz/style.py, so the figure matches the other paper figures.
LC_LABEL_SIZE  = 22
LC_TICK_SIZE   = 21
LC_LEGEND_SIZE = 18
LC_VALUE_SIZE  = 13
LC_COLORS = {                       # seaborn "deep", as in viz/style.py
    'EnergyLens (Ours)': '#4c72b0',
    'MaverIQ (Lat×P)':   '#dd8452',
    'Gradient Boosting': '#55a868',
    'Random Forest':     '#c44e52',
    'Linear Regression': '#8172b3',
}
LC_MARKERS = {                      # filled markers: white edges as in viz/style.py
    'EnergyLens (Ours)': 'o',
    'MaverIQ (Lat×P)':   's',
    'Gradient Boosting': 'D',
    'Random Forest':     '^',
    'Linear Regression': 'v',
}


def plot_learning_summary(records, output_dir):
    """
    Median held-out MAPE across datasets per method and n_train, IQR shaded,
    over the n_train values every dataset reaches. Also prints on how many
    datasets Gradient Boosting ever overtakes EnergyLens.
    """
    # Linear Regression (80-130% MAPE) is left out: it would flatten the gap that matters.
    r = records[records.method.isin(LC_COLORS) & (records.method != 'Linear Regression')]
    n_ds = r.dataset.nunique()
    ns = sorted(n for n, g in r.groupby('n_train') if g.dataset.nunique() == n_ds)
    r = r[r.n_train.isin(ns)]
    plt.ioff()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, method in enumerate([m for m in LC_COLORS if m in set(r.method)]):
        g = r[r.method == method].groupby('n_train').mape
        med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
        color = LC_COLORS[method]
        ax.plot(ns, med.loc[ns], label=method, color=color, marker=LC_MARKERS.get(method, 'o'),
                markersize=9, linewidth=2.2, markeredgecolor='white', markeredgewidth=0.8,
                ls='-' if method in ('EnergyLens (Ours)', 'MaverIQ (Lat×P)') else '--')
        ax.fill_between(ns, q1.loc[ns], q3.loc[ns], color=color, alpha=0.12, lw=0)
    ax.set_xlabel('Training samples (n_train)', fontsize=LC_LABEL_SIZE)
    ax.set_ylabel('MAPE (%)', fontsize=LC_LABEL_SIZE)
    ax.set_xscale('log')
    labelled = []
    for n in ns:
        if not labelled or n / labelled[-1] >= 1.25:
            labelled.append(n)
    ax.set_xticks(labelled)
    ax.set_xticks([n for n in ns if n not in labelled], minor=True)
    ax.set_xticklabels([str(n) for n in labelled], fontsize=LC_TICK_SIZE, rotation=30, ha='right')
    ax.tick_params(axis='y', labelsize=LC_TICK_SIZE)
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor('#f9f9f9')
    ax.legend(fontsize=LC_LEGEND_SIZE, framealpha=0.9, loc='best')
    fig.tight_layout()
    out = os.path.join(output_dir, 'learning_curve_summary.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out}")

    p = records.pivot_table(index=['dataset', 'n_train'], columns='method', values='mape').reset_index()
    el, gb = 'EnergyLens (Ours)', 'Gradient Boosting'
    if el in p and gb in p:
        first = {ds: next((n for n in sorted(g.n_train) if g.loc[g.n_train == n, gb].iloc[0]
                           < g.loc[g.n_train == n, el].iloc[0]), None) for ds, g in p.groupby('dataset')}
        over = {ds: n for ds, n in first.items() if n is not None}
        print(f"  Gradient Boosting overtakes EnergyLens on {len(over)}/{len(first)} datasets"
              + (f" (first n_train: {over})" if over else ""))


def plot_learning_curves(all_results, output_dir, all_stds=None):
    """
    Line plot: for each dataset, MAPE vs n_train for all methods, over the full
    range of training sizes. Shades ±1σ (across seeds) and annotates the
    crossover point where Gradient Boosting overtakes EnergyLens, so the
    small-sample advantage claim is backed by the whole curve, not two points.
    No title (the caption carries it); styled like viz/style.py.
    Saves one figure per dataset.
    """
    all_stds = all_stds or {}
    plt.ioff()
    for ds, results_by_n in all_results.items():
        n_vals  = sorted(results_by_n.keys())
        methods = list(next(iter(results_by_n.values())).keys())
        stds_by_n = all_stds.get(ds, {})
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for i, method in enumerate(methods):
            color = LC_COLORS.get(method, f'C{i}')
            mapes = np.array([results_by_n[n].get(method, (np.nan,))[0] for n in n_vals])
            ax.plot(n_vals, mapes, label=method, color=color,
                    marker=LC_MARKERS.get(method, 'o'), markersize=9, linewidth=2.2,
                    markeredgecolor='white', markeredgewidth=0.8,
                    ls='-' if method in ('EnergyLens (Ours)', 'MaverIQ (Lat×P)') else '--')
            # ±1σ band across seeds
            sds = np.array([stds_by_n.get(n, {}).get(method, np.nan) for n in n_vals])
            if np.isfinite(sds).any():
                ax.fill_between(n_vals, mapes - sds, mapes + sds,
                                color=color, alpha=0.12, lw=0)

        ax.set_xlabel('Training samples (n_train)', fontsize=LC_LABEL_SIZE)
        ax.set_ylabel('MAPE (%)', fontsize=LC_LABEL_SIZE)
        ax.set_xscale('log')
        # Label a tick only if it is far enough (x1.25) from the last labelled
        # one: on a log axis 500/600/700 collide at 15 pt. Unlabelled sizes
        # keep their markers and a minor tick.
        labelled = []
        for n in n_vals:
            if not labelled or n / labelled[-1] >= 1.25:
                labelled.append(n)
        ax.set_xticks(labelled)
        ax.set_xticks([n for n in n_vals if n not in labelled], minor=True)
        ax.set_xticklabels([str(n) for n in labelled], fontsize=LC_TICK_SIZE,
                           rotation=30, ha='right')
        ax.tick_params(axis='y', labelsize=LC_TICK_SIZE)
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_facecolor('#f9f9f9')

        # Annotate where Gradient Boosting overtakes EnergyLens (the crossover).
        # After the axis setup so the label sits inside the final y-limits.
        ours = np.array([results_by_n[n].get('EnergyLens (Ours)', (np.nan,))[0] for n in n_vals])
        gb   = np.array([results_by_n[n].get('Gradient Boosting', (np.nan,))[0] for n in n_vals])
        diff = ours - gb   # >0 means GB is better (lower MAPE)
        cross = np.where((diff[:-1] <= 0) & (diff[1:] > 0))[0]
        if len(cross):
            i = cross[0]
            # linear interpolation of the crossover n_train
            x0, x1 = n_vals[i], n_vals[i + 1]
            d0, d1 = diff[i], diff[i + 1]
            xc = x0 + (x1 - x0) * (-d0) / (d1 - d0) if d1 != d0 else x0
            ax.axvline(xc, color='grey', ls=':', lw=1.2)
            # Left of the line, at the top: the legend sits on the right and
            # the curves converge low, so this corner is free.
            ax.annotate(f'crossover ≈ n={xc:.0f}', xy=(xc, ax.get_ylim()[1]),
                        xytext=(-6, -6), textcoords='offset points',
                        ha='right', va='top', fontsize=LC_VALUE_SIZE, color='dimgrey')

        ax.legend(fontsize=LC_LEGEND_SIZE, framealpha=0.9, loc='best')
        fig.tight_layout()
        slug = ds.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        out  = os.path.join(output_dir, f'learning_curve_{slug}.png')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {out}")


# ════════════════════════════════════════════════════════════
#  PRINT TABLE — matches paper format
# ════════════════════════════════════════════════════════════

def print_table_1(results_by_dataset, n50, n500):
    """Table 1: MAPE / R² / RMSE at n_train=50 and n_train=500 per dataset."""
    methods = ['EnergyLens (Ours)', 'Linear Regression', 'Random Forest',
               'Gradient Boosting', 'MaverIQ (Lat×P)']
    w_col = 24

    def fmt(mape, r2, rmse):
        return f"{mape:5.1f}% / {r2:6.3f}  {rmse:7.2f}"

    hdr_m = f"{'MAPE / R²':^22}  {'RMSE':^7}"
    print(f"\n{'='*110}")
    print(f"  TABLE 1 — MAPE / R² / RMSE at n_train={n50} and n_train={n500}")
    print(f"{'='*110}")
    print(f"  {'Method':<{w_col}}  {'n_train='+str(n50):^32}  {'n_train='+str(n500):^32}")
    print(f"  {'':<{w_col}}  {hdr_m}  {hdr_m}")
    print(f"  {'-'*100}")

    for ds, by_n in results_by_dataset.items():
        print(f"\n  ── {ds} ──")
        for m in methods:
            r50  = by_n.get(n50,  {}).get(m, (np.nan, np.nan, np.nan))
            r500 = by_n.get(n500, {}).get(m, (np.nan, np.nan, np.nan))
            s50  = fmt(*r50)  if not np.isnan(r50[0])  else '  n/a            n/a'
            s500 = fmt(*r500) if not np.isnan(r500[0]) else '  n/a            n/a'
            print(f"  {m:<{w_col}}  {s50}  {s500}")


def print_train_vs_heldout(records):
    """Train vs held-out MAPE per method and n_train, averaged over datasets (overfitting check)."""
    t = records.groupby(['n_train', 'method'])[['train_mape', 'mape', 'gap']].mean().unstack('method')
    methods = ['EnergyLens (Ours)', 'Gradient Boosting', 'Random Forest',
               'Linear Regression', 'MaverIQ (Lat×P)']
    methods = [m for m in methods if m in t['mape'].columns]
    print(f"\n{'='*110}")
    print("  TRAIN vs HELD-OUT MAPE (mean over datasets) — train / held-out / gap, in %")
    print(f"{'='*110}")
    print(f"  {'n_train':>7}" + "".join(f"{m[:22]:>26}" for m in methods))
    for n in t.index:
        cells = "".join(f"{t.loc[n, ('train_mape', m)]:>8.1f} /{t.loc[n, ('mape', m)]:>6.1f} /{t.loc[n, ('gap', m)]:>6.1f}"
                        for m in methods)
        print(f"  {n:>7}{cells}")


def print_table_2(results_by_dataset, n_train):
    """Table 2: MAPE / R² at n_train=50 across all datasets (cross-model)."""
    methods = ['EnergyLens (Ours)', 'Linear Regression', 'Random Forest',
               'Gradient Boosting', 'MaverIQ (Lat×P)']

    col_w = 16
    header = "  " + f"{'Dataset':<30}" + "".join(f"{m:^{col_w}}" for m in methods)
    print(f"\n{'='*110}")
    print(f"  TABLE 2 — MAPE / R² at n_train={n_train} across hardware/model combinations")
    print(f"{'='*110}")
    print(header)
    print(f"  {'':<30}" + "".join(f"{'MAPE / R²':^{col_w}}" for _ in methods))
    print(f"  {'-'*100}")

    for ds, by_n in results_by_dataset.items():
        by_m = by_n.get(n_train, {})
        row = f"  {ds:<30}"
        for m in methods:
            mape, r2, _ = by_m.get(m, (np.nan, np.nan, np.nan))
            if np.isnan(mape):
                cell = f"{'n/a':^{col_w}}"
            else:
                cell = f"{mape:5.1f}%/{r2:5.3f}".center(col_w)
            row += cell
        print(row)


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset_args import add_datasets_arg, resolve_datasets
    ap = argparse.ArgumentParser(
        description="Cross-model comparison: EnergyLens vs ML baselines vs MaverIQ")
    ap.add_argument('--output-dir',  default='cross_model_comparison',
                    help='Directory for plots and CSV outputs')
    ap.add_argument('--n-train',     type=int, nargs='+', default=N_TRAIN_VALS,
                    help='Training-set sizes for the learning curve '
                         '(default: full sweep 25..700)')
    ap.add_argument('--test-frac',   type=float, default=TEST_FRAC,
                    help='Fraction of rows held out per seed for scoring '
                         '(default 0.2). 0 = score on the full dataset, '
                         'training rows included (old behaviour).')
    ap.add_argument('--seeds',       type=int, default=N_SEEDS,
                    help='Number of random seeds to average over (default: 3)')
    ap.add_argument('--summary-from', default=None,
                    help='Only redraw learning_curve_summary.png from an existing '
                         'cross_model_results.csv (no fitting)')
    ap.add_argument('--no-plots',    action='store_true',
                    help='Skip plot generation')
    add_datasets_arg(ap)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.summary_from:
        plot_learning_summary(pd.read_csv(args.summary_from), args.output_dir)
        return

    datasets = resolve_datasets(args.datasets, DATASETS)

    missing = [(n, p) for n, p in datasets.items() if not os.path.isfile(p)]
    if missing:
        print("[WARNING] Missing datasets:")
        for n, p in missing:
            print(f"  {n}: {p}")
        datasets = {k: v for k, v in datasets.items() if (k, v) not in missing}
    if not datasets:
        print("[ERROR] No datasets available. Exiting."); return

    # all_results[dataset][n_train][method] = (mape, r2, rmse)
    # all_stds[dataset][n_train][method]    = mape std across seeds
    all_results = {}
    all_stds    = {}
    row_records = []

    for ds_name, ds_path in datasets.items():
        print(f"\n{'━'*70}")
        print(f"  Dataset: {ds_name}")
        print(f"  Path:    {ds_path}")
        raw = pd.read_csv(ds_path)
        print(f"  Rows:    {len(raw)}")
        try:
            df = prepare_df(raw, ds_path)
        except ValueError as e:
            print(f"  [SKIP] {e}"); continue

        all_results[ds_name] = {}
        all_stds[ds_name]    = {}

        for n_train in sorted(args.n_train):
            if n_train > len(df):
                print(f"  [SKIP] n_train={n_train} > dataset size {len(df)}")
                continue
            print(f"\n  ── n_train={n_train} (averaging over {args.seeds} seeds) ──")
            out = evaluate_at_n(df, n_train, seeds=args.seeds, test_frac=args.test_frac)
            if out is None:
                pool = len(df) - max(1, int(round(args.test_frac * len(df))))
                print(f"  [SKIP] n_train={n_train} > training pool {pool} "
                      f"(dataset {len(df)} minus held-out test rows)")
                continue
            by_method, by_std, by_train, by_time = out
            all_results[ds_name][n_train] = by_method
            all_stds[ds_name][n_train]    = by_std

            for method, (mape, r2, rmse) in by_method.items():
                train = by_train.get(method, np.nan)
                fit_s = by_time.get(method, np.nan)
                print(f"    {method:<28}  MAPE={mape:6.1f}%  (train {train:6.1f}%)  R²={r2:.4f}  RMSE={rmse:.3f}  fit={fit_s:.3f}s")
                row_records.append({
                    'dataset': ds_name, 'n_train': n_train, 'method': method,
                    'mape': round(mape, 3), 'r2': round(r2, 4), 'rmse': round(rmse, 4),
                    'mape_std': round(by_std.get(method, np.nan), 3),
                    'train_mape': round(train, 3), 'gap': round(mape - train, 3),
                    'fit_time_s': round(fit_s, 4),
                })

    # ── Save CSV
    if row_records:
        csv_path = os.path.join(args.output_dir, 'cross_model_results.csv')
        pd.DataFrame(row_records).to_csv(csv_path, index=False)
        print(f"\n  Saved results CSV → {csv_path}")

    # ── Print summary tables
    # Use the sizes that actually ran (some may be skipped as > training pool)
    n_train_vals = sorted({n for by_n in all_results.values() for n in by_n}) or sorted(args.n_train)
    n50  = n_train_vals[0]
    n500 = n_train_vals[-1]
    print_table_1(all_results, n50, n500)
    print_table_2(all_results, n50)
    if row_records:
        print_train_vs_heldout(pd.DataFrame(row_records))

    # ── Plots
    if not args.no_plots and all_results:
        print(f"\n  Generating plots → {args.output_dir}/")
        for n_train in sorted(args.n_train):
            by_ds = {ds: results[n_train]
                     for ds, results in all_results.items()
                     if n_train in results}
            if by_ds:
                plot_mape_bar(by_ds, n_train, args.output_dir)
        plot_mape_heatmap(all_results, args.output_dir)
        plot_learning_curves(all_results, args.output_dir, all_stds)
        if row_records:
            plot_learning_summary(pd.DataFrame(row_records), args.output_dir)

    print(f"\n{'━'*70}")
    print("  Done.")


if __name__ == '__main__':
    main()
