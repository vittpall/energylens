#!/usr/bin/env python3
"""Pairwise Ranking Evaluation for Energy Config Selection

    python pairwise_ranking_eval.py
    python pairwise_ranking_eval.py --output-dir ranking_results
    python pairwise_ranking_eval.py --unified-n 50
    python pairwise_ranking_eval.py --loco        # add leave-one-config-out rows
"""

# ════════════════════════════════════════════════════════════
#  DATASETS — same as compare_configs.py
# ════════════════════════════════════════════════════════════
DATASETS = {
    # "Mistral-7B":              "./results/run_text_only_Mistral-7B-Instruct-v0.3/profiling_metrics.csv",
    # "NVIDIA-Nemotron-V2-9B":   "./results/run_text_only_NVIDIA-Nemotron-Nano-9B-v2/profiling_metrics.csv",
    # "Qwen1.5-MoE-2.7B":       "./results/run_text_only_Qwen1.5-MoE-A2.7B/profiling_metrics.csv",
    # "Qwen2-VL-7B-text":       "./results/run_text_only_Qwen2-VL-7B-Instruct/profiling_metrics.csv",
    # "Qwen2-VL-7B-image-chat": "./results/run_text_image_Qwen2-VL-7B-Instruct/profiling_metrics.csv",
    # "Qwen2-VL-7B-video-chat": "./results/run_text_video_Qwen2-VL-7B-Instruct/combined_metrics.csv",
    # "Qwen2-VL-7B-video-only": "./results/run_video_only_Qwen2-VL-7B-Instruct/profiling_metrics.csv",
    # "LLava-1.5-7B-video-chat": "./results/run_text_video_llava-1.5-7b-hf/combined_metrics.csv",
    # "LLava-1.5-7B-image-chat": "./results/run_text_image_llava-1.5-7b-hf/profiling_metrics.csv",
    # "Qwen2.5-32B-text" : "./results/run_text_only_Qwen2.5-32B-Instruct/res_native/profiling_metrics.csv",
    # "Qwen2.5-14B-text" : "./results/run_text_only_Qwen2.5-14B-Instruct/res_native/profiling_metrics.csv",
    # "Mixtral-8x7B-Instruct-v0.1" : "./results/run_text_only_Mixtral-8x7B-Instruct-v0.1/res_native/profiling_metrics.csv",
    #"InternVL2-Llama-76B" : "./results/run_text_image_InternVL2-Llama3-76B/res_native/profiling_metrics.csv"
    "Qwen2.5-14B-fp16 (zeus)":  "./results_zeus/run_text_only_Qwen2.5-14B-Instruct_fp16/res_native/profiling_metrics.csv",
    "Qwen2.5-14B-text":         "./results/run_text_only_Qwen2.5-14B-Instruct/res_native/profiling_metrics.csv",
    "Qwen2.5-14B-bnb4bit":      "./results/run_text_only_Qwen2.5-14B-Instruct_fp16-bnb4bit/res_native/profiling_metrics.csv",
}

import argparse, sys, os, warnings, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_args import add_datasets_arg, resolve_datasets
from compare_cross_model import (drop_invalid_rows, max_num_seqs_column, effective_batch, vision_tokens,
                                 total_input_tokens, decode_tokens)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize as opt
from scipy.stats import spearmanr, kendalltau
warnings.filterwarnings('ignore')


# ════════════════════════════════════════════════════════════
#  COLUMN MAPPING (reused from compare_configs.py)
# ════════════════════════════════════════════════════════════

def prepare_df(df, data_path):
    """Map column names to internal names."""
    df = drop_invalid_rows(df, data_path)
    out = pd.DataFrame()

    required_cols = {
        'latency': 'measured_latency',
        'energy': 'measured_energy',
        'power': 'gpu_power_w',
        'tensor_parallel_size': 'tp_degree',
        'pipeline_parallel_size': 'pp_degree',
        'batch_size': 'batch_size'
    }
    for src, dst in required_cols.items():
        if src not in df.columns:
            raise ValueError(f"Required column '{src}' not found in {data_path}")
        dtype = int if dst in ['tp_degree', 'pp_degree', 'batch_size'] else float
        out[dst] = df[src].astype(dtype)

    if 'max_tokens' in df.columns:
        out['max_tokens'] = df['max_tokens'].astype(int)
    elif 'output_length' in df.columns:
        out['max_tokens'] = df['output_length'].astype(int)
    elif 'output_tokens' in df.columns:
        out['max_tokens'] = df['output_tokens'].astype(int)
    else:
        out['max_tokens'] = 1

    out['input_length'] = df['input_length'].astype(int) if 'input_length' in df.columns else 0

    out['image_tokens'], out['video_tokens'] = vision_tokens(df)
    out['total_input_tokens'] = total_input_tokens(
        out['input_length'], np.maximum(out['image_tokens'], out['video_tokens']),
        out['batch_size'], data_path)
    out['max_num_seqs'] = max_num_seqs_column(df, data_path)
    out['decode_tokens'] = decode_tokens(df, out['max_tokens'])

    # Failed-generation rows have blank latency/power/energy (the profiler writes
    # empty cells on a generation exception), which become NaN here. Drop them so
    # ranking metrics aren't computed against missing energies.
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
#  MODEL A: EnergyLens (Prefill/Decode Split)
# ════════════════════════════════════════════════════════════

def unified_predict(p, df):
    α_p, β_p, γ1_p, γ2_p, ε_p, α_d, β_d, γ1_d, γ2_d, ε_d, δ1, δ2 = p
    bs  = effective_batch(df)
    tok = df['decode_tokens'].values.astype(float)
    tp  = df['tp_degree'].values.astype(float)
    pp  = df['pp_degree'].values.astype(float)
    inp = df['total_input_tokens'].values.astype(float)
    E_prefill = α_p * inp / (np.power(bs, β_p) + ε_p) * np.power(tp, γ1_p) * np.power(pp, γ2_p)
    E_decode  = α_d * tok / (np.power(bs, β_d) + ε_d) * np.power(tp, γ1_d) * np.power(pp, γ2_d)
    return E_prefill + E_decode + δ1 * tp + δ2 * pp


def fit_unified(df, restarts=20):
    true = df['measured_energy'].values.astype(float)
    bs   = effective_batch(df)
    tok  = df['decode_tokens'].values.astype(float)
    tp   = df['tp_degree'].values.astype(float)
    pp   = df['pp_degree'].values.astype(float)
    inp  = df['total_input_tokens'].values.astype(float)
    mask = true > 0
    med  = np.median(true[mask]) if mask.any() else 1.0
    med_tok = np.median(tok[mask]) if mask.any() else 1.0
    med_inp = np.median(inp[mask]) if mask.any() else 1.0

    def loss(p):
        α_p, β_p, γ1_p, γ2_p, ε_p, α_d, β_d, γ1_d, γ2_d, ε_d, δ1, δ2 = p
        E_pf = α_p * inp / (np.power(bs, β_p) + ε_p) * np.power(tp, γ1_p) * np.power(pp, γ2_p)
        E_dc = α_d * tok / (np.power(bs, β_d) + ε_d) * np.power(tp, γ1_d) * np.power(pp, γ2_d)
        pred = E_pf + E_dc + δ1 * tp + δ2 * pp
        return np.mean(np.abs(pred[mask] - true[mask]) / true[mask])

    scale_p = med * 0.3 / max(med_inp, 1)
    scale_d = med * 0.7 / max(med_tok, 1)

    best, best_p = np.inf, None
    for i in range(restarts):
        rng = np.random.RandomState(i * 42)
        x0 = [
            scale_p * rng.uniform(0.1, 3.0), rng.uniform(0.3, 1.5),
            rng.uniform(-1.2, -0.2), rng.uniform(-1.0, 0.0), rng.uniform(0.01, 2.0),
            scale_d * rng.uniform(0.1, 3.0), rng.uniform(0.3, 1.5),
            rng.uniform(-1.2, -0.2), rng.uniform(-1.0, 0.0), rng.uniform(0.01, 2.0),
            rng.uniform(-med*0.05, med*0.1), rng.uniform(-med*0.05, med*0.1),
        ]
        bounds = [
            (1e-8,1e6),(0.01,3),(-2,3),(-2,3),(0,100),
            (1e-8,1e6),(0.01,3),(-2,3),(-2,3),(0,100),
            (-1e6,1e6),(-1e6,1e6),
        ]
        r = opt(loss, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 10000, 'ftol': 1e-15})
        if r.fun < best:
            best, best_p = r.fun, r.x
    return best_p


# ════════════════════════════════════════════════════════════
#  MODEL B: MaverIQ (Latency × Power)
# ════════════════════════════════════════════════════════════

def fit_maveriq(df):
    tp  = df['tp_degree'].values.astype(float)
    pp  = df['pp_degree'].values.astype(float)
    bs  = df['batch_size'].values.astype(float)
    tok = df['decode_tokens'].values.astype(float)
    true_lat = df['measured_latency'].values.astype(float)
    mask = true_lat > 0

    def predict_lat(p, tp_, pp_, bs_, tok_):
        L11t, L11p, Ot_t, Ot_p, Op_t, Op_p, Obs, a, b, g, d = p
        def comp(L11, Ot, Op):
            t1 = L11 / (tp_ * np.power(np.maximum(pp_, 1), a))
            t2 = np.where(tp_ > 1, np.power(tp_-1, b)/np.power(np.maximum(pp_,1), g)*Ot, 0.0)
            t3 = np.where(pp_ > 1, np.power(pp_-1, d)*Op, 0.0)
            return t1 + t2 + t3
        ttft = comp(L11t, Ot_t, Op_t)
        tpot = comp(L11p, Ot_p, Op_p)
        return np.maximum(ttft + tok_ * tpot + np.maximum(bs_-1, 0) * Obs, 1e-6)

    def loss(p):
        pred = predict_lat(p, tp, pp, bs, tok)
        return np.mean(np.abs(pred[mask] - true_lat[mask]) / true_lat[mask])

    ml = np.median(true_lat[mask]) if mask.any() else 1.0
    mt = max(np.median(tok), 1)
    x0 = [ml*0.3, ml*0.7/mt, 0.01, 0.005, 0.005, 0.002, 0.02, 0.85, 1.15, 0.45, 0.70]
    bounds = [(1e-6,100),(1e-6,10),(0,10),(0,5),(0,10),(0,5),(0,10),(0.1,2),(0.1,3),(0.01,2),(0.1,2)]
    best, best_p = np.inf, x0
    for i in range(3):
        rng = np.random.RandomState(i * 17)
        x0r = [v*rng.uniform(0.5,2) for v in x0[:7]] + list(x0[7:])
        x0r = [np.clip(x0r[j], bounds[j][0], bounds[j][1]) for j in range(len(bounds))]
        r = opt(loss, x0r, method='L-BFGS-B', bounds=bounds, options={'maxiter':5000,'ftol':1e-12})
        if r.fun < best: best, best_p = r.fun, r.x

    def lat_fn(df2):
        return predict_lat(best_p,
            df2['tp_degree'].values.astype(float), df2['pp_degree'].values.astype(float),
            df2['batch_size'].values.astype(float), df2['decode_tokens'].values.astype(float))
    return best_p, lat_fn


def _latency_features(df):
    bs  = df['batch_size'].values.astype(float)
    eff = effective_batch(df)
    return (df['total_input_tokens'].values.astype(float), df['decode_tokens'].values.astype(float),
            eff, np.ceil(bs / eff), df['tp_degree'].values.astype(float),
            df['pp_degree'].values.astype(float))


def _latency_model(p, inp, tok, eff, waves, tp, pp):
    a_p, g1_p, g2_p, a_d, b_d, β, g1_d, g2_d, c_tp, c_pp = p
    prefill = a_p * inp * np.power(tp, g1_p) * np.power(pp, g2_p)
    decode  = waves * tok * (a_d + b_d * np.power(eff, β)) * np.power(tp, g1_d) * np.power(pp, g2_d)
    return prefill + decode + c_tp * tp + c_pp * pp


def fit_latency(df, restarts=20):
    """Batch latency for the SLO feasibility check; returns lat_fn(df).

    Unlike MaverIQ's latency model it has a prefill term (input tokens), and a
    batch above max_num_seqs runs in ceil(bs / max_num_seqs) decode waves.
    Fit on |log(pred / measured)|, which converges more reliably than MAPE.
    """
    X = _latency_features(df)
    y = df['measured_latency'].values.astype(float)
    m = y > 0
    ml = np.median(y[m])
    mi, mt = max(np.median(X[0]), 1), max(np.median(X[1]), 1)

    def loss(p):
        return np.mean(np.abs(np.log(np.maximum(_latency_model(p, *X)[m], 1e-9) / y[m])))

    bounds = [(0, 10), (-2, 2), (-2, 2), (1e-9, 10), (0, 10), (0.01, 2), (-2, 2), (-2, 2),
              (-100, 100), (-100, 100)]
    best, best_p = np.inf, None
    for i in range(restarts):
        rng = np.random.RandomState(i)
        x0 = [ml * 0.2 / mi * rng.uniform(0.2, 3), rng.uniform(-1, 0), rng.uniform(-0.5, 0.5),
              ml * 0.6 / mt * rng.uniform(0.2, 3), ml * 0.1 / mt * rng.uniform(0, 2),
              rng.uniform(0.3, 1.2), rng.uniform(-1, 0), rng.uniform(-0.5, 0.5), 0, 0]
        r = opt(loss, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 10000})
        if r.fun < best:
            best, best_p = r.fun, r.x
    return lambda df2: np.maximum(_latency_model(best_p, *_latency_features(df2)), 1e-6)


def maveriq_energy(lat_fn, df):
    # `measured_latency` is per batch but `measured_energy` is per request, so
    # latency x power yields whole-batch energy and must be divided by the batch
    # size to match the target. Verified on the profiling CSVs:
    # power * latency / batch_size reproduces the logged energy to <=1% MAPE.
    # Omitting the divisor inflates MAPE by a factor of ~batch_size (thousands
    # of percent); it does not affect within-scenario ranking, where batch_size
    # is held fixed.
    bs = np.maximum(df['batch_size'].values.astype(float), 1.0)
    return lat_fn(df) * df['gpu_power_w'].values / bs


# ════════════════════════════════════════════════════════════
#  MODEL C: Gradient Boosting (generic ML baseline)
# ════════════════════════════════════════════════════════════
# Same regressor, hyper-parameters and features as compare_cross_model.py's
# ML baselines, so the ranking results line up with the MAPE comparison.
# MaverIQ is the prior system; Gradient Boosting is the "just fit a model"
# baseline — together they separate "better than the state of the art" from
# "the physical structure of the formula is what helps".
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    _SKLEARN = True
except ImportError:
    _SKLEARN = False
    print("[WARNING] scikit-learn not found – Gradient Boosting will be skipped.")


def _gb_features(df):
    vis = np.maximum(df['image_tokens'].values.astype(float),
                     df['video_tokens'].values.astype(float))
    return np.column_stack([
        df['batch_size'].values.astype(float),
        df['decode_tokens'].values.astype(float),
        df['tp_degree'].values.astype(float),
        df['pp_degree'].values.astype(float),
        df['total_input_tokens'].values.astype(float),
        vis,
    ])


def fit_gb(train_df):
    """Fit Gradient Boosting on energy; returns a predict(df) function."""
    X = _gb_features(train_df)
    sc = StandardScaler().fit(X)
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(sc.transform(X), train_df['measured_energy'].values.astype(float))
    return lambda df: gb.predict(sc.transform(_gb_features(df)))


# ════════════════════════════════════════════════════════════
#  LEAVE-ONE-CONFIGURATION-OUT (LOCO)
# ════════════════════════════════════════════════════════════

def loco_predictions(df, fit_pool=None):
    """
    Leave-one-configuration-out energy predictions.

    For each unique (TP,PP) configuration c, every model is fit on every row of
    `fit_pool` whose config is NOT c, then used to predict the rows of `df` whose
    config IS c — a configuration neither model saw during fitting. This is what
    turns "row-savings" (fewer profiling rows per config) into "reload-savings"
    (skipping the profiling of an entire configuration): if the ranking still
    holds for a held-out config, that config never needs to be profiled/reloaded.

    `fit_pool` defaults to `df`. When a scenario-level train/test split is in
    effect, pass the train rows as `fit_pool` so a held-out config is predicted
    by a model blind to BOTH that config and the test scenarios.

    Returns a dict {method_label: pred_array} where each array is aligned
    positionally to df's rows and every entry was produced by a model blind to
    that row's own configuration. Entries stay NaN if a config could not be
    held out (e.g. it is the only configuration present in the fit pool).
    """
    if fit_pool is None:
        fit_pool = df
    n = len(df)
    uni = np.full(n, np.nan)
    mav = np.full(n, np.nan)
    gbp = np.full(n, np.nan)
    configs = df[['tp_degree', 'pp_degree']].drop_duplicates().values

    if fit_pool[['tp_degree', 'pp_degree']].drop_duplicates().shape[0] < 2:
        print("    [LOCO] fewer than two configurations in the fit pool — skipping")
        return {}

    skipped = []
    for tp_c, pp_c in configs:
        pred_mask  = (df['tp_degree'] == tp_c) & (df['pp_degree'] == pp_c)
        train_df   = fit_pool[(fit_pool['tp_degree'] != tp_c) | (fit_pool['pp_degree'] != pp_c)]
        pred_df    = df[pred_mask]
        if len(train_df) < 5 or len(pred_df) == 0:
            continue   # not enough data to fit without this config; leave NaN
        remaining = train_df[['tp_degree', 'pp_degree']].drop_duplicates().values
        if not _loco_identifiable((tp_c, pp_c), remaining):
            skipped.append((int(tp_c), int(pp_c)))
            continue   # held-out config not identifiable from the rest; leave NaN
        # EnergyLens: fit on the other configs, predict the held-out one
        try:
            p = fit_unified(train_df)
            uni[pred_mask.values] = unified_predict(p, pred_df)
        except Exception as e:
            print(f"    [LOCO] EnergyLens failed for config (TP={tp_c},PP={pp_c}): {e}")
        # MaverIQ: same holdout
        try:
            _, lat_fn = fit_maveriq(train_df)
            mav[pred_mask.values] = maveriq_energy(lat_fn, pred_df)
        except Exception as e:
            print(f"    [LOCO] MaverIQ failed for config (TP={tp_c},PP={pp_c}): {e}")
        # Gradient Boosting: same holdout. Trees cannot extrapolate to a TP/PP
        # value absent from training — they return the nearest seen region.
        if _SKLEARN:
            try:
                gbp[pred_mask.values] = fit_gb(train_df)(pred_df)
            except Exception as e:
                print(f"    [LOCO] Gradient Boosting failed for config (TP={tp_c},PP={pp_c}): {e}")

    if skipped:
        print(f"    [LOCO] skipped {len(skipped)}/{len(configs)} configurations whose "
              f"(TP,PP) the remaining configs cannot identify: {skipped}")
    if len(skipped) == len(configs):
        print("    [LOCO] no identifiable configuration — no LOCO rows for this dataset")
        return {}
    print(f"    [LOCO] held out {len(configs) - len(skipped)} configurations "
          f"(EnergyLens covered {np.isfinite(uni).sum()}/{n} rows, "
          f"MaverIQ {np.isfinite(mav).sum()}/{n}, "
          f"Gradient Boosting {np.isfinite(gbp).sum()}/{n})")
    out = {'EnergyLens (LOCO)': uni, 'MaverIQ (LOCO)': mav}
    if _SKLEARN:
        out['Gradient Boosting (LOCO)'] = gbp
    return out


def _loco_identifiable(held, remaining):
    """
    True if configuration `held` = (TP,PP) can be predicted from `remaining`
    (the configs left in the training set) without guessing a parallelism
    effect that was never varied.

    Both models are (close to) log-linear in TP and PP, so the check is whether
    held's feature vector [1, log TP, log PP] lies in the span of the remaining
    configs' vectors. Examples:
      - TP∈{1,2}, PP=1 only (AMD MI100): holding out TP=2 leaves only TP=1, so
        the TP direction was never observed → not identifiable.
      - the standard 6-config grid: the other 5 vary both TP and PP → fine.
      - configs on one line TP×PP=const (Mixtral, InternVL2): the held-out
        point lies on the line the others span → identifiable (moving along an
        observed direction), even though TP and PP cannot be separated.
    """
    A = np.array([[1.0, np.log(t), np.log(p)] for t, p in remaining]).T   # 3 × k
    x = np.array([1.0, np.log(held[0]), np.log(held[1])])
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return bool(np.allclose(A @ coef, x, atol=1e-8))


# ════════════════════════════════════════════════════════════
#  PAIRWISE RANKING EVALUATION
# ════════════════════════════════════════════════════════════

def compute_pairwise_metrics(true_energy, pred_energy_dict, config_labels):
    """
    Given true energy and predicted energy for each method across configs
    within a single scenario, compute pairwise ranking metrics.

    Args:
        true_energy: array of measured energy per config (length = n_configs)
        pred_energy_dict: {'method_name': array of predicted energy per config}
        config_labels: list of (TP,PP) tuples for each config

    Returns:
        dict of metrics per method
    """
    n = len(true_energy)
    if n < 2:
        return {}

    results = {}
    for method, pred in pred_energy_dict.items():
        # Pairwise accuracy: for every pair (i,j), did we predict the right order?
        correct = 0
        total = 0
        ties_true = 0
        for i in range(n):
            for j in range(i+1, n):
                true_diff = true_energy[i] - true_energy[j]
                pred_diff = pred[i] - pred[j]

                if abs(true_diff) < 1e-9:
                    ties_true += 1
                    continue  # skip true ties

                total += 1
                # Correct if both diffs have the same sign
                if true_diff * pred_diff > 0:
                    correct += 1
                # If pred_diff == 0 (tie in prediction), count as 0.5
                elif abs(pred_diff) < 1e-9:
                    correct += 0.5

        pairwise_acc = correct / total if total > 0 else np.nan

        # Spearman rank correlation
        if np.std(true_energy) > 1e-9 and np.std(pred) > 1e-9:
            spearman_rho, spearman_p = spearmanr(true_energy, pred)
            kendall_tau, kendall_p = kendalltau(true_energy, pred)
        else:
            spearman_rho = np.nan
            kendall_tau = np.nan

        # Top-1 selection
        true_best_idx = np.argmin(true_energy)
        pred_best_idx = np.argmin(pred)
        top1_correct = (true_best_idx == pred_best_idx)

        # Top-1 regret
        oracle_e = true_energy[true_best_idx]
        selected_e = true_energy[pred_best_idx]
        regret_pct = (selected_e - oracle_e) / oracle_e * 100 if oracle_e > 0 else 0

        # Top-2 accuracy: is the true best in the model's top-2 picks?
        pred_top2 = np.argsort(pred)[:2]
        top2_correct = true_best_idx in pred_top2

        results[method] = {
            'pairwise_acc': pairwise_acc,
            'spearman_rho': spearman_rho,
            'kendall_tau': kendall_tau,
            'top1_correct': top1_correct,
            'top2_correct': top2_correct,
            'regret_pct': regret_pct,
            'n_pairs': total,
            'n_configs': n,
            'oracle_config': config_labels[true_best_idx],
            'selected_config': config_labels[pred_best_idx],
        }

    return results


# ════════════════════════════════════════════════════════════
#  PLOTTING
# ════════════════════════════════════════════════════════════

METHOD_COLORS = {
    'EnergyLens':          '#2a9d8f',
    'MaverIQ':             '#e76f51',
    'Gradient Boosting':   '#f4a261',
    'EnergyLens (LOCO)':   '#1d6f63',
    'Gradient Boosting (LOCO)': '#c77d3a',
    'MaverIQ (LOCO)':      '#b5462f',
}
_METHOD_ORDER = ['EnergyLens', 'MaverIQ', 'Gradient Boosting',
                 'EnergyLens (LOCO)', 'MaverIQ (LOCO)', 'Gradient Boosting (LOCO)']


def _methods_in_order(all_results):
    present = set(r['method'] for r in all_results)
    return [m for m in _METHOD_ORDER if m in present] + \
           [m for m in sorted(present) if m not in _METHOD_ORDER]


def plot_pairwise_summary(all_results, output_dir):
    """Bar chart of pairwise accuracy and Spearman rho per dataset per method."""
    plt.ioff()

    # Aggregate by dataset
    datasets = sorted(set(r['dataset'] for r in all_results))
    methods = _methods_in_order(all_results)

    # --- Pairwise accuracy ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metric, title, ylabel in [
        (axes[0], 'pairwise_acc', 'Pairwise Ranking Accuracy', 'Accuracy (%)'),
        (axes[1], 'spearman_rho', 'Spearman Rank Correlation', 'Spearman ρ'),
    ]:
        x = np.arange(len(datasets))
        width = 0.8 / max(len(methods), 1)

        for mi, method in enumerate(methods):
            vals = []
            for ds in datasets:
                ds_results = [r for r in all_results if r['dataset'] == ds and r['method'] == method]
                if metric == 'pairwise_acc':
                    v = np.mean([r['pairwise_acc'] for r in ds_results]) * 100
                else:
                    v = np.mean([r['spearman_rho'] for r in ds_results if not np.isnan(r['spearman_rho'])])
                vals.append(v)

            offset = (mi - (len(methods) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width,
                         label=method, color=METHOD_COLORS.get(method, f'C{mi}'),
                         edgecolor='black', linewidth=0.5)

            for bar, v in zip(bars, vals):
                fmt = f'{v:.1f}%' if metric == 'pairwise_acc' else f'{v:.3f}'
                ax.annotate(fmt,
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=7, fontweight='bold')

        if metric == 'pairwise_acc':
            ax.axhline(50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
        elif metric == 'spearman_rho':
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylim(-0.5, 1.1)

        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=9)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    fig.tight_layout()
    out = os.path.join(output_dir, 'pairwise_ranking_summary.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved → {out}")


def plot_regret_distribution(all_results, output_dir):
    """Histogram of regret distribution for each method."""
    plt.ioff()

    methods = _methods_in_order(all_results)

    fig, ax = plt.subplots(figsize=(10, 5))
    for mi, method in enumerate(methods):
        regrets = [r['regret_pct'] for r in all_results if r['method'] == method]
        ax.hist(regrets, bins=30, alpha=0.6, label=method,
                color=METHOD_COLORS.get(method, f'C{mi}'), edgecolor='black')

    ax.set_xlabel('Top-1 Regret (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count (scenarios)', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Config Selection Regret', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.axvline(5, color='black', linestyle=':', alpha=0.5, label='5% threshold')

    fig.tight_layout()
    out = os.path.join(output_dir, 'regret_distribution.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out}")


# ════════════════════════════════════════════════════════════
#  MAIN EVALUATION LOOP
# ════════════════════════════════════════════════════════════

def process_dataset(data_path, output_dir, unified_n, dataset_name, loco=False,
                    test_frac=0.3, split_seed=42):
    raw = pd.read_csv(data_path)
    df = prepare_df(raw, data_path)

    print(f"\n{'='*70}")
    print(f"  {dataset_name}: {len(df)} rows")
    print(f"  TP: {sorted(df['tp_degree'].unique())}  PP: {sorted(df['pp_degree'].unique())}")
    print(f"  BS: {sorted(df['batch_size'].unique())}  MaxTok: {sorted(df['max_tokens'].unique())}")
    print(f"{'='*70}")

    # ── Scenario-level train/test split ──────────────────────────────────
    # Fairness: (#1) all models fit on the SAME rows, and (#2) ranking is
    # scored only on held-out scenarios neither model saw while fitting. We
    # split on whole scenarios (unique max_tokens×batch_size workload points),
    # not rows, because there is ~1 row per (scenario,config) cell — a row-level
    # split would strip configs out of scenarios and destroy the ranking.
    scenarios = list(df.groupby(['max_tokens', 'batch_size']).groups.keys())
    n_scen = len(scenarios)
    scen_key = list(zip(df['max_tokens'], df['batch_size']))

    if test_frac and test_frac > 0 and n_scen >= 4:
        rng = np.random.RandomState(split_seed)
        perm = rng.permutation(n_scen)
        n_test = max(1, int(round(test_frac * n_scen)))
        n_test = min(n_test, n_scen - 2)            # keep ≥2 train scenarios
        eval_keys = set(scenarios[i] for i in perm[:n_test])
        eval_mask = np.array([k in eval_keys for k in scen_key])
        train_df  = df[~eval_mask]
        split_desc = (f"scenario holdout — {n_test}/{n_scen} test scenarios "
                      f"(seed {split_seed})")
    else:
        if test_frac and test_frac > 0:
            print(f"  [split] only {n_scen} scenarios — too few to hold out; "
                  f"falling back to IN-SAMPLE (not a fair generalization test)")
        eval_keys = None                            # evaluate on every scenario
        eval_mask = np.ones(len(df), dtype=bool)
        train_df  = df
        split_desc = "in-sample (fit and evaluate on all rows)"

    # Optional cap on the SHARED training rows (fed identically to every model)
    if unified_n > 0 and unified_n < len(train_df):
        rng2 = np.random.RandomState(split_seed + 1)
        cap_idx = rng2.choice(len(train_df), size=unified_n, replace=False)
        train_df = train_df.iloc[cap_idx]

    print(f"  Split: {split_desc}")
    print(f"  Training rows (shared by all models): {len(train_df)}")

    # Fit BOTH models on the identical training rows (fairness #1)
    uni_params    = fit_unified(train_df)
    _, mav_lat_fn = fit_maveriq(train_df)

    # Predict over the full df; the scenario loop below scores only held-out
    # scenarios (fairness #2). LOCO fits on the same train pool.
    uni_pred = unified_predict(uni_params, df)
    mav_pred = maveriq_energy(mav_lat_fn, df)
    gb_pred  = fit_gb(train_df)(df) if _SKLEARN else None   # same rows (fairness #1)
    true_e   = df['measured_energy'].values

    # Reference MAPE, on the evaluation rows only
    te = true_e[eval_mask]
    uni_mape = np.mean(np.abs(uni_pred[eval_mask] - te) / np.maximum(te, 1e-6)) * 100
    mav_mape = np.mean(np.abs(mav_pred[eval_mask] - te) / np.maximum(te, 1e-6)) * 100
    gb_str = ""
    if gb_pred is not None:
        gb_mape = np.mean(np.abs(gb_pred[eval_mask] - te) / np.maximum(te, 1e-6)) * 100
        gb_str = f"  Gradient Boosting: {gb_mape:.1f}%"
    print(f"  Energy MAPE (eval rows) → EnergyLens: {uni_mape:.1f}%  MaverIQ: {mav_mape:.1f}%{gb_str}")

    # Prediction vectors, each aligned positionally to df's rows. The in-sample
    # methods keep their original labels; LOCO methods are added only when asked.
    pred_arrays = {
        'EnergyLens': uni_pred,
        'MaverIQ':    mav_pred,
    }
    if gb_pred is not None:
        pred_arrays['Gradient Boosting'] = gb_pred
    if loco:
        print(f"  Computing leave-one-configuration-out predictions ...")
        pred_arrays.update(loco_predictions(df, fit_pool=train_df))

    # ── Pairwise evaluation, restricted to held-out scenarios ──
    scenario_results = []

    for (tok, bs), group in df.groupby(['max_tokens', 'batch_size']):
        if eval_keys is not None and (tok, bs) not in eval_keys:
            continue
        if len(group) < 2:
            continue

        # Aggregate to mean energy per (TP,PP) config within this scenario
        config_means = group.groupby(['tp_degree', 'pp_degree']).agg(
            true_energy=('measured_energy', 'mean'),
        ).reset_index()
        if len(config_means) < 2:
            continue

        config_labels = [(r['tp_degree'], r['pp_degree']) for _, r in config_means.iterrows()]
        true_arr      = config_means['true_energy'].values.astype(float)

        # Per-config mean of each method's prediction (same aggregation for all)
        pred_dict = {}
        for method, arr in pred_arrays.items():
            vals = []
            for _, row in config_means.iterrows():
                mask = (group['tp_degree'] == row['tp_degree']) & (group['pp_degree'] == row['pp_degree'])
                sub_idx = group[mask].index
                vals.append(np.mean(arr[sub_idx]))
            vals = np.array(vals)
            # A LOCO method may leave a config unpredicted (NaN) if it couldn't be
            # held out; skip that method for this scenario rather than corrupt it.
            if np.isnan(vals).any():
                continue
            pred_dict[method] = vals

        metrics = compute_pairwise_metrics(true_arr, pred_dict, config_labels)

        for method, m in metrics.items():
            scenario_results.append({
                'dataset': dataset_name,
                'method': method,
                'max_tokens': tok,
                'batch_size': bs,
                **m,
            })

    return scenario_results


def main():
    ap = argparse.ArgumentParser(description='Pairwise Ranking Evaluation')
    ap.add_argument('--output-dir', type=str, default='results_figures')
    ap.add_argument('--unified-n', type=int, default=0,
                    help='Cap the shared training-row count (0 = use all train '
                         'rows). Both models are fit on the same sampled rows.')
    ap.add_argument('--test-frac', type=float, default=0.3,
                    help='Fraction of SCENARIOS held out for evaluation. Both '
                         'models fit on the train scenarios and are scored only '
                         'on held-out ones (fair generalization test). '
                         'Use 0 to reproduce the old in-sample behavior.')
    ap.add_argument('--split-seed', type=int, nargs='+', default=[42],
                    help='Seed(s) for the scenario split and the --unified-n draw. '
                         'Several seeds are pooled; the summary also reports '
                         'the spread across seeds.')
    ap.add_argument('--loco', action='store_true',
                    help='Also run leave-one-configuration-out: fit each model on '
                         'all configs but one and rank the held-out config. Adds '
                         '"EnergyLens (LOCO)" / "MaverIQ (LOCO)" / '
                         '"Gradient Boosting (LOCO)" rows.')
    add_datasets_arg(ap)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Check which datasets exist
    datasets  = resolve_datasets(args.datasets, DATASETS)
    available = {k: v for k, v in datasets.items() if os.path.isfile(v)}
    missing = {k: v for k, v in datasets.items() if not os.path.isfile(v)}
    if missing:
        print(f"Skipping {len(missing)} missing dataset(s): {list(missing.keys())}")
    if not available:
        print("No datasets found. Edit the DATASETS dictionary.")
        sys.exit(1)

    all_results = []

    for seed in args.split_seed:
        for dataset_name, data_path in available.items():
            try:
                results = process_dataset(data_path, args.output_dir, args.unified_n,
                                           dataset_name, loco=args.loco,
                                           test_frac=args.test_frac,
                                           split_seed=seed)
                all_results.extend({**r, 'seed': seed} for r in results)
            except Exception as e:
                print(f"  ERROR processing {dataset_name} (seed {seed}): {e}")
                import traceback; traceback.print_exc()

    if not all_results:
        print("No results generated.")
        sys.exit(1)

    # ── Per-dataset summary table ──
    print(f"\n{'='*110}")
    print(f"  PAIRWISE RANKING RESULTS")
    print(f"{'='*110}")

    print(f"\n  {'Dataset':<25} {'Method':<18} {'PairAcc':>8} {'Spearman':>9} "
          f"{'Kendall':>8} {'Top1%':>6} {'Top2%':>6} {'MeanReg':>8} {'<5%Reg':>7} {'#Scen':>6}")
    print(f"  {'─'*104}")

    datasets = sorted(set(r['dataset'] for r in all_results))
    summary_rows = []

    # Method display order: in-sample first, then LOCO, then anything unexpected.
    methods_present = _methods_in_order(all_results)

    for ds in datasets:
        for method in methods_present:
            rows = [r for r in all_results if r['dataset'] == ds and r['method'] == method]
            if not rows:
                continue

            pair_acc  = np.mean([r['pairwise_acc'] for r in rows]) * 100
            spear     = np.mean([r['spearman_rho'] for r in rows if not np.isnan(r['spearman_rho'])])
            kendall   = np.mean([r['kendall_tau'] for r in rows if not np.isnan(r['kendall_tau'])])
            top1      = np.mean([r['top1_correct'] for r in rows]) * 100
            top2      = np.mean([r['top2_correct'] for r in rows]) * 100
            mean_reg  = np.mean([r['regret_pct'] for r in rows])
            lt5_reg   = np.mean([r['regret_pct'] < 5 for r in rows]) * 100
            n_scen    = len(rows)

            print(f"  {ds:<25} {method:<18} {pair_acc:>7.1f}% {spear:>8.3f} "
                  f"{kendall:>8.3f} {top1:>5.1f}% {top2:>5.1f}% {mean_reg:>7.2f}% {lt5_reg:>6.1f}% {n_scen:>6}")

            summary_rows.append({
                'dataset': ds, 'method': method,
                'pairwise_acc': round(pair_acc, 2),
                'spearman_rho': round(spear, 4),
                'kendall_tau': round(kendall, 4),
                'top1_acc': round(top1, 2),
                'top2_acc': round(top2, 2),
                'mean_regret_pct': round(mean_reg, 2),
                'lt5_regret_pct': round(lt5_reg, 2),
                'n_scenarios': n_scen,
            })

    # ── Overall aggregation ──
    print(f"\n  {'─'*104}")
    for method in methods_present:
        rows = [r for r in all_results if r['method'] == method]
        if not rows:
            continue
        pair_acc  = np.mean([r['pairwise_acc'] for r in rows]) * 100
        spear     = np.mean([r['spearman_rho'] for r in rows if not np.isnan(r['spearman_rho'])])
        kendall   = np.mean([r['kendall_tau'] for r in rows if not np.isnan(r['kendall_tau'])])
        top1      = np.mean([r['top1_correct'] for r in rows]) * 100
        top2      = np.mean([r['top2_correct'] for r in rows]) * 100
        mean_reg  = np.mean([r['regret_pct'] for r in rows])
        lt5_reg   = np.mean([r['regret_pct'] < 5 for r in rows]) * 100
        n_scen    = len(rows)

        print(f"  {'OVERALL':<25} {method:<18} {pair_acc:>7.1f}% {spear:>8.3f} "
              f"{kendall:>8.3f} {top1:>5.1f}% {top2:>5.1f}% {mean_reg:>7.2f}% {lt5_reg:>6.1f}% {n_scen:>6}")

    # ── Spread across seeds (overall means per seed) ──
    detail_df = pd.DataFrame(all_results)
    if detail_df['seed'].nunique() > 1:
        per_seed = detail_df.groupby(['method', 'seed']).agg(
            pairwise=('pairwise_acc', lambda v: 100 * v.mean()),
            top1=('top1_correct', lambda v: 100 * v.mean()),
            regret=('regret_pct', 'mean')).groupby('method').agg(['mean', 'std'])
        print(f"\n  OVERALL across {detail_df['seed'].nunique()} seeds (mean ± std of per-seed means)")
        for m in methods_present:
            if m in per_seed.index:
                r = per_seed.loc[m]
                print(f"  {m:<26} pairwise {r[('pairwise','mean')]:5.1f} ± {r[('pairwise','std')]:4.1f}   "
                      f"top1 {r[('top1','mean')]:5.1f} ± {r[('top1','std')]:4.1f}   "
                      f"regret {r[('regret','mean')]:5.2f} ± {r[('regret','std')]:4.2f}")

    # ── Save CSVs ──
    summary_df = pd.DataFrame(summary_rows)

    detail_path = os.path.join(args.output_dir, 'pairwise_detail.csv')
    summary_path = os.path.join(args.output_dir, 'pairwise_summary.csv')
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Saved → {detail_path}")
    print(f"  Saved → {summary_path}")

    # ── Plots ──
    plot_pairwise_summary(all_results, args.output_dir)
    plot_regret_distribution(all_results, args.output_dir)


if __name__ == '__main__':
    main()