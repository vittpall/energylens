import numpy as np
import pandas as pd
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os
import json
import argparse

try:
    import sympy
    _HAS_SYMPY = True
except ImportError:
    _HAS_SYMPY = False


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def load_dataset(path):
    df_all = None
    for f in sorted(os.listdir(path)):
        if f.endswith(".csv"):
            file_path = os.path.join(path, f)
            pdf = pd.read_csv(file_path)
            print(f"  Loaded {file_path} with shape {pdf.shape}")
            df_all = pdf if df_all is None else pd.concat([df_all, pdf], ignore_index=True)
    if df_all is None:
        raise ValueError(f"No CSV files found in {path}")
    return df_all


def sample_per_config(df, feature_cols, samples_per_config=10, random_state=42):
    """
    Sample up to `samples_per_config` rows for each unique combination of feature values.
    This balances the dataset and reduces the impact of repeated measurements.
    """
    # Create a tuple key for each unique configuration
    df = df.copy()
    
    # Only use feature columns that exist in the dataframe
    existing_cols = [c for c in feature_cols if c in df.columns]
    
    if not existing_cols:
        print(f"  Warning: No feature columns found for grouping, returning original df")
        return df
    
    # Group by unique feature configurations and sample
    sampled_dfs = []
    grouped = df.groupby(existing_cols, dropna=False)
    
    n_configs = len(grouped)
    n_original = len(df)
    
    for config_values, group in grouped:
        if len(group) <= samples_per_config:
            sampled_dfs.append(group)
        else:
            sampled_dfs.append(group.sample(n=samples_per_config, random_state=random_state))
    
    df_sampled = pd.concat(sampled_dfs, ignore_index=True)
    
    # Shuffle the result
    df_sampled = df_sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    print(f"  Sampling: {n_original} rows -> {len(df_sampled)} rows "
          f"({n_configs} unique configs, up to {samples_per_config} samples each)")
    
    return df_sampled


def engineer_features(df):
    df = df.copy()
    if "prompt" in df.columns:
        df = df.drop(columns=["prompt"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    df["parallelism"] = df["tensor_parallel_size"] * df["pipeline_parallel_size"]
    df["img_x_input"] = df["image_resolution"] * df["input_length"]
    if "batch_size" in df.columns and "input_length" in df.columns:
        df["total_input_tokens"] = df["input_length"] * df["batch_size"]
        df["tokens_per_gpu"] = df["total_input_tokens"] / df["parallelism"]
        df["workload_per_gpu"] = df["total_input_tokens"] / df["parallelism"]
    if "batch_size" in df.columns and "max_num_batched_tokens" in df.columns:
        df["batch_utilization"] = df["batch_size"] / df["max_num_batched_tokens"]
    df["img_pixels"] = df["image_resolution"] ** 2
    df["log_aspect"] = np.abs(np.log(df["aspect_ratio"].clip(lower=1e-8)))

    if "dtype" in df.columns:
        df = pd.get_dummies(df, columns=["dtype"], drop_first=False)

    return df


BASE_FEATURE_COLS = [
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "parallelism",
    "input_length",
    #"image_resolution",
    #"aspect_ratio",
    #"log_aspect",
    "batch_size",
    "max_num_batched_tokens",
    #"max_num_seqs",
    "total_input_tokens",
    #"tokens_per_gpu",
    #"batch_utilization",
    #"img_x_input",
    #"img_pixels",
    #"workload_per_gpu",
    "max_tokens",
    #"gpu_memory_utilization",
]

# Columns to use for defining unique configurations (subset of features that define a "config")
# These should be the controllable/experimental parameters, not derived features
CONFIG_COLS = [
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "batch_size",
    "max_num_batched_tokens",
    "max_num_seqs",
    "max_tokens",
    "image_resolution",
]


def get_feature_cols(df):
    cat_cols = [c for c in df.columns if c.startswith("dtype_")]
    return [c for c in BASE_FEATURE_COLS + cat_cols if c in df.columns]


def get_config_cols(df):
    """Get the columns that define a unique experimental configuration."""
    return [c for c in CONFIG_COLS if c in df.columns]


def save_hall_of_fame(model, output_dir, y_mean, y_std, target_name, approach_name, dataset_name):
    try:
        equations_df = model.equations_.copy()
        equations_df["y_mean"] = y_mean
        equations_df["y_std"] = y_std
        equations_df["target"] = target_name
        equations_df["approach"] = approach_name
        equations_df["dataset"] = dataset_name

        csv_path = os.path.join(output_dir, f"hall_of_fame_{dataset_name}_{approach_name}.csv")
        equations_df.to_csv(csv_path, index=False)
        print(f"    Hall of Fame saved to {csv_path}")

        top_equations = []
        for idx in range(min(10, len(equations_df))):
            eq_row = equations_df.iloc[idx]
            top_equations.append({
                "rank": int(idx + 1),
                "equation": str(eq_row["equation"]),
                "complexity": int(eq_row["complexity"]),
                "loss": float(eq_row["loss"]),
                "score": float(eq_row["score"]) if "score" in eq_row else None,
            })

        json_path = os.path.join(output_dir, f"hall_of_fame_{dataset_name}_{approach_name}_top10.json")
        with open(json_path, "w") as f:
            json.dump({
                "dataset": dataset_name,
                "y_mean": y_mean,
                "y_std": y_std,
                "target": target_name,
                "approach": approach_name,
                "top_equations": top_equations,
            }, f, indent=2)
        print(f"    Top 10 equations saved to {json_path}")
        return csv_path, json_path

    except Exception as e:
        print(f"    Warning: Could not save Hall of Fame: {e}")
        import traceback; traceback.print_exc()
        return None, None


def train_pysr(X_tr, y_tr, var_names, target_name, output_dir,
               niterations=2000, maxsize=25, parsimony=1e-4, approach_name=None,
               random_state=42):
    y_mean = float(y_tr.mean())
    y_std = float(y_tr.std()) if y_tr.std() != 0 else 1.0
    y_scaled = (y_tr - y_mean) / y_std

    temp_filename = f"temp_pysr_{approach_name or target_name}_seed{random_state}.csv"
    temp_equation_file = os.path.join(output_dir, temp_filename)

    model = PySRRegressor(
        niterations=niterations,
        population_size=200,
        maxsize=maxsize,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "log", "abs", "neg"],
        loss="loss(x, y) = abs(x - y)",
        complexity_of_constants=1,
        model_selection="best",
        parsimony=parsimony,
        verbosity=1,
        random_state=random_state,
        temp_equation_file=temp_equation_file,
    )
    model.fit(X_tr, y_scaled, variable_names=var_names)

    try:
        if os.path.exists(temp_equation_file):
            os.remove(temp_equation_file)
    except:
        pass

    return model, y_mean, y_std


def predict_unscaled(model, X, y_mean, y_std):
    pred = model.predict(X)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    return pred * y_std + y_mean


def evaluate(y_true, y_pred, label):
    y_pred_clean = np.nan_to_num(y_pred,
                                  nan=np.nanmean(y_true),
                                  posinf=np.nanmax(y_true),
                                  neginf=np.nanmin(y_true))
    mae = mean_absolute_error(y_true, y_pred_clean)
    r2 = r2_score(y_true, y_pred_clean)
    mape = np.mean(np.abs((y_true - y_pred_clean) /
                   np.clip(np.abs(y_true), 1e-8, None))) * 100
    print(f"  {label} - MAE: {mae:.4f}, R²: {r2:.4f}, MAPE: {mape:.1f}%")
    return {"mae": float(mae), "r2": float(r2), "mape": float(mape)}


# ──────────────────────────────────────────────
# Approach A — Direct energy prediction
# ──────────────────────────────────────────────
def run_approach_A(df, feature_cols, output_dir, dataset_name, niterations=2000, maxsize=25):
    os.makedirs(output_dir, exist_ok=True)

    X = df[feature_cols].values.astype(np.float64)
    y_energy = df["energy"].values.astype(np.float64)

    idx_train, idx_test = train_test_split(
        np.arange(len(X)), test_size=0.3, random_state=42
    )
    X_train, X_test = X[idx_train], X[idx_test]

    print("\n" + "=" * 50)
    print("APPROACH A: Direct energy prediction")
    print("=" * 50)

    model, y_mean, y_std = train_pysr(
        X_train, y_energy[idx_train], feature_cols, "energy_direct", output_dir,
        niterations=niterations, maxsize=maxsize, approach_name="A_direct"
    )

    hof_csv, hof_json = save_hall_of_fame(
        model, output_dir, y_mean, y_std, "energy", "A_direct", dataset_name
    )

    pred_train = predict_unscaled(model, X_train, y_mean, y_std)
    pred_test = predict_unscaled(model, X_test, y_mean, y_std)
    metrics_train = evaluate(y_energy[idx_train], pred_train, "Train")
    metrics_test = evaluate(y_energy[idx_test], pred_test, "Test")
    best_eq = str(model.get_best()["equation"])
    print(f"\n  Best equation: {best_eq}")

    results = {
        "feature_cols": feature_cols,
        "approach": "A_direct",
        "equation": best_eq,
        "y_mean": y_mean,
        "y_std": y_std,
        "target": "energy",
        "train_metrics": metrics_train,
        "test_metrics": metrics_test,
        "hall_of_fame_csv": hof_csv,
        "hall_of_fame_json": hof_json,
    }

    results_path = os.path.join(output_dir, "equations.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    return results


# ──────────────────────────────────────────────
# Motif-stability seed sweep
# ──────────────────────────────────────────────
#
# Reviewer R2: confirm the structural motifs in the EnergyLens formula are
# reproducible across runs, not cherry-picked from one lucky search. We hold the
# (engineered, sampled) training data fixed and vary ONLY the PySR random seed,
# then classify each seed's selected "best" equation into named structural
# motifs. How often each motif recurs across N seeds is direct evidence that the
# structure is discovered, not imposed.
#
# Motif → paper mapping:
#   parallelism_in_denominator -> "token load over parallelism"
#   batch_in_denominator       -> "batch-size saturation"
#   log_compression            -> recurring log(...) compressed-ratio form
#   tp_and_pp_both             -> "TP/PP asymmetry" (both axes appear)

# Variables (from BASE_FEATURE_COLS) that represent parallelism / batching.
_PARALLELISM_VARS = {"tensor_parallel_size", "pipeline_parallel_size", "parallelism"}
_BATCH_VARS = {"batch_size"}

_MOTIF_NAMES = [
    "log_compression",
    "sqrt",
    "square",
    "parallelism_in_denominator",
    "batch_in_denominator",
    "tp_and_pp_both",
]


def _equation_to_sympy(model):
    """Return the model-selected best equation as a sympy expression, or None."""
    if not _HAS_SYMPY:
        return None
    try:
        expr = model.sympy()
        if isinstance(expr, (list, tuple)):
            expr = expr[0]
        return sympy.sympify(expr)
    except Exception:
        return None


def classify_motifs(expr):
    """Classify a sympy expression into named structural motifs (bool flags)."""
    if expr is None:
        return {m: False for m in _MOTIF_NAMES}

    present = {s.name for s in expr.free_symbols}

    has_log = bool(expr.has(sympy.log))
    powers = list(expr.atoms(sympy.Pow))
    has_sqrt = any(p.exp == sympy.Rational(1, 2) for p in powers)
    has_square = any(p.exp == 2 for p in powers)

    try:
        _num, den = sympy.fraction(sympy.together(expr))
        den_syms = {s.name for s in den.free_symbols}
    except Exception:
        den_syms = set()

    return {
        "log_compression": has_log,
        "sqrt": has_sqrt,
        "square": has_square,
        "parallelism_in_denominator": bool(den_syms & _PARALLELISM_VARS),
        "batch_in_denominator": bool(den_syms & _BATCH_VARS),
        "tp_and_pp_both": {"tensor_parallel_size", "pipeline_parallel_size"}.issubset(present),
    }


def classify_motifs_front(model, top_k=None):
    """Classify motifs across the whole Pareto front (hall of fame), not just the
    single selected best equation.

    A motif counts as present for this seed if ANY equation in the front contains
    it. We also record the *minimum complexity* at which each motif first appears,
    which shows how deep in the search the structure emerges (e.g. TP/PP
    separation typically appears only above the trivial reduced forms).

    top_k: if set, restrict to the top_k equations by PySR 'score' (marginal
    accuracy gain); default None scans the entire front.

    Returns (flags, first_complexity).
    """
    flags = {m: False for m in _MOTIF_NAMES}
    first_cx = {m: None for m in _MOTIF_NAMES}
    if not _HAS_SYMPY:
        return flags, first_cx
    try:
        eqs = model.equations_.copy()
    except Exception:
        return flags, first_cx

    if top_k and "score" in eqs.columns:
        keep = eqs.sort_values("score", ascending=False).head(top_k).index
        eqs = eqs.loc[keep]
    # Scan simplest → most complex so first_cx is the simplest form with the motif
    eqs = eqs.sort_values("complexity")

    for idx, row in eqs.iterrows():
        expr = row.get("sympy_format", None)
        if expr is None:
            try:
                expr = model.sympy(index=idx)
            except Exception:
                continue
        try:
            expr = sympy.sympify(expr)
        except Exception:
            continue
        f = classify_motifs(expr)
        cx = int(row["complexity"]) if "complexity" in row else None
        for m in _MOTIF_NAMES:
            if f[m]:
                flags[m] = True
                if first_cx[m] is None:
                    first_cx[m] = cx
    return flags, first_cx


def _plot_motif_frequency(freq, n_seeds, output_dir, dataset_name):
    """Bar chart of motif recurrence fraction across seeds."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    [WARN] matplotlib not available — skipping motif plot.")
        return

    motifs = list(freq.keys())
    vals = [freq[m] * 100 for m in motifs]

    fig, ax = plt.subplots(figsize=(max(7, len(motifs) * 1.3), 4.5))
    bars = ax.bar(range(len(motifs)), vals, color="#2a9d8f",
                  edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.0f}%", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(motifs)))
    ax.set_xticklabels(motifs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Recurrence across seeds (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.axhline(100, color="gray", ls="--", alpha=0.4)
    ax.set_title(f"Motif Reproducibility — {dataset_name} (n={n_seeds} seeds)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", ls="--", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(output_dir, f"motif_frequency_{dataset_name}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved motif plot → {out}")


def run_seed_sweep(df, feature_cols, output_dir, dataset_name,
                   n_seeds=10, niterations=2000, maxsize=25,
                   resample_per_seed=False, base_seed=42, pareto_top_k=None):
    """Run PySR over N seeds and report structural-motif reproducibility.

    Mirrors Approach A (direct energy prediction) but loops over seeds. By
    default the train/test split is fixed (base_seed) so only the PySR search
    varies; --resample-per-seed also redraws the split for a stricter test.

    Motifs are detected across the whole Pareto front (not just the selected
    best equation), so the structure of the full discovered formula family is
    assessed — the single parsimony-selected best is often a reduced form that
    collapses, e.g., TP/PP into one term. pareto_top_k optionally limits the
    scan to the top-k equations by score (None = entire front).
    """
    os.makedirs(output_dir, exist_ok=True)
    if not _HAS_SYMPY:
        print("    [WARN] sympy not available — motif columns will be empty.")

    X = df[feature_cols].values.astype(np.float64)
    y = df["energy"].values.astype(np.float64)

    idx0_train, idx0_test = train_test_split(
        np.arange(len(X)), test_size=0.3, random_state=base_seed
    )

    print("\n" + "=" * 50)
    print(f"MOTIF-STABILITY SEED SWEEP — {dataset_name} ({n_seeds} seeds)")
    print(f"  resample_per_seed={resample_per_seed}  "
          f"niterations={niterations}  maxsize={maxsize}")
    print("=" * 50)

    per_seed_rows = []
    motif_records = []
    first_cx_records = []

    for seed in range(n_seeds):
        if resample_per_seed:
            idx_train, idx_test = train_test_split(
                np.arange(len(X)), test_size=0.3, random_state=seed
            )
        else:
            idx_train, idx_test = idx0_train, idx0_test

        model, y_mean, y_std = train_pysr(
            X[idx_train], y[idx_train], feature_cols, "energy_direct", output_dir,
            niterations=niterations, maxsize=maxsize,
            approach_name=f"seedsweep_{dataset_name}", random_state=seed,
        )

        best = model.get_best()
        best_eq = str(best["equation"])
        complexity = int(best["complexity"])
        expr = _equation_to_sympy(model)

        # Motifs across the full Pareto front (not just the selected best)
        motifs, first_cx = classify_motifs_front(model, top_k=pareto_top_k)

        pred_test = predict_unscaled(model, X[idx_test], y_mean, y_std)
        m = evaluate(y[idx_test], pred_test, f"seed {seed} Test")
        motif_str = ", ".join(
            f"{k}@cx{first_cx[k]}" for k, v in motifs.items() if v) or "(none)"
        print(f"    best eq (cx{complexity}): {best_eq}")
        print(f"    front motifs: {motif_str}")

        per_seed_rows.append({
            "dataset": dataset_name,
            "seed": seed,
            "mape": round(m["mape"], 3),
            "r2": round(m["r2"], 4),
            "best_complexity": complexity,
            "best_equation": best_eq,
            "best_sympy": str(expr) if expr is not None else "",
            **{f"motif_{k}": int(v) for k, v in motifs.items()},
            **{f"firstcx_{k}": (first_cx[k] if first_cx[k] is not None else "")
               for k in _MOTIF_NAMES},
        })
        motif_records.append(motifs)
        first_cx_records.append(first_cx)

    freq = {
        mname: sum(rec[mname] for rec in motif_records) / len(motif_records)
        for mname in _MOTIF_NAMES
    }

    def _median_first_cx(mname):
        vals = [rec[mname] for rec in first_cx_records if rec[mname] is not None]
        return float(np.median(vals)) if vals else None

    print(f"\n{'─'*68}")
    print(f"  MOTIF FREQUENCY across Pareto front — {dataset_name} ({n_seeds} seeds)")
    print(f"  {'motif':<32}  {'count':>8}  {'fraction':>9}  {'median cx':>10}")
    print(f"  {'-'*66}")
    for mname in _MOTIF_NAMES:
        cnt = sum(rec[mname] for rec in motif_records)
        mcx = _median_first_cx(mname)
        mcx_str = f"{mcx:.0f}" if mcx is not None else "—"
        print(f"  {mname:<32}  {cnt:>4}/{n_seeds:<3}  {freq[mname]:>8.0%}  {mcx_str:>10}")

    mapes = [r["mape"] for r in per_seed_rows]
    print(f"\n  Test MAPE across seeds: mean={np.mean(mapes):.2f}%  "
          f"std={np.std(mapes):.2f}%  min={np.min(mapes):.2f}%  max={np.max(mapes):.2f}%")

    seed_csv = os.path.join(output_dir, f"seed_equations_{dataset_name}.csv")
    pd.DataFrame(per_seed_rows).to_csv(seed_csv, index=False)
    print(f"\n  Saved per-seed equations → {seed_csv}")

    freq_csv = os.path.join(output_dir, f"motif_frequency_{dataset_name}.csv")
    pd.DataFrame(
        [{"motif": mname, "count": int(sum(rec[mname] for rec in motif_records)),
          "n_seeds": n_seeds, "fraction": round(freq[mname], 4)}
         for mname in _MOTIF_NAMES]
    ).to_csv(freq_csv, index=False)
    print(f"  Saved motif frequency → {freq_csv}")

    _plot_motif_frequency(freq, n_seeds, output_dir, dataset_name)

    return {
        "dataset": dataset_name,
        "n_seeds": n_seeds,
        "motif_frequency": freq,
        "mape_mean": float(np.mean(mapes)),
        "mape_std": float(np.std(mapes)),
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Symbolic Regression — Approach A: direct energy")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Dataset folders, e.g. data/llava data/qwen2vl")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--niterations", type=int, default=2000)
    parser.add_argument("--maxsize", type=int, default=25)
    parser.add_argument("--samples-per-config", type=int, default=10,
                        help="Number of samples to keep per unique feature configuration (default: 10)")
    parser.add_argument("--no-sampling", action="store_true",
                        help="Disable per-config sampling, use all data")
    parser.add_argument("--seed-sweep", type=int, default=0, metavar="N",
                        help="Run PySR over N seeds and report structural-motif "
                             "reproducibility instead of a single fit. 0 = off.")
    parser.add_argument("--resample-per-seed", action="store_true",
                        help="Also redraw the train/test split per seed during "
                             "the sweep (stricter joint data+search stability).")
    parser.add_argument("--pareto-top-k", type=int, default=0, metavar="K",
                        help="Scan only the top-K Pareto equations by score for "
                             "motifs (0 = scan the entire front, default).")
    args = parser.parse_args()

    all_results = {}
    sweep_results = {}

    for dataset_path in args.datasets:
        dataset_name = os.path.basename(os.path.normpath(dataset_path))
        output_dir = os.path.join(args.output_root, dataset_name)

        print("\n" + "#" * 70)
        print(f"# DATASET: {dataset_name} ({dataset_path})")
        print("#" * 70)

        df_raw = load_dataset(dataset_path)
        df = engineer_features(df_raw)
        feature_cols = get_feature_cols(df)
        config_cols = get_config_cols(df)

        print(f"  Rows (after feature engineering): {len(df)}")
        print(f"  Features: {len(feature_cols)} -> {feature_cols}")
        print(f"  Config columns for grouping: {config_cols}")

        # Apply per-config sampling
        if not args.no_sampling:
            df = sample_per_config(df, config_cols, 
                                   samples_per_config=args.samples_per_config,
                                   random_state=42)
        else:
            print(f"  Sampling disabled, using all {len(df)} rows")

        print(f"  Final rows: {len(df)}")
        print(f"  Output: {output_dir}")

        if args.seed_sweep > 0:
            sweep_results[dataset_name] = run_seed_sweep(
                df, feature_cols, output_dir, dataset_name,
                n_seeds=args.seed_sweep,
                niterations=args.niterations, maxsize=args.maxsize,
                resample_per_seed=args.resample_per_seed,
                pareto_top_k=(args.pareto_top_k or None),
            )
            continue

        results = run_approach_A(
            df, feature_cols, output_dir, dataset_name,
            niterations=args.niterations, maxsize=args.maxsize
        )
        all_results[dataset_name] = results["test_metrics"]

    if sweep_results:
        print("\n" + "#" * 70)
        print("# MOTIF-STABILITY SUMMARY (recurrence across seeds)")
        print("#" * 70)
        header = f"  {'Dataset':<25} {'MAPE%(mean±std)':>18}  " + \
                 "  ".join(m[:10] for m in _MOTIF_NAMES)
        print(header)
        print("  " + "-" * (len(header)))
        for ds_name, s in sweep_results.items():
            freq_str = "  ".join(f"{s['motif_frequency'][m]*100:>9.0f}%" for m in _MOTIF_NAMES)
            print(f"  {ds_name:<25} {s['mape_mean']:>8.1f}±{s['mape_std']:<6.1f}  {freq_str}")
        summary_path = os.path.join(args.output_root, "motif_stability_summary.json")
        with open(summary_path, "w") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"\n  Saved → {summary_path}")

    if len(args.datasets) > 1 and all_results:
        print("\n" + "#" * 70)
        print("# CROSS-DATASET SUMMARY (Test set)")
        print("#" * 70)
        print(f"  {'Dataset':<25} {'MAE':>10} {'R²':>10} {'MAPE%':>8}")
        print("  " + "-" * 55)
        for ds_name, m in all_results.items():
            print(f"  {ds_name:<25} {m['mae']:>10.4f} {m['r2']:>10.4f} {m['mape']:>7.1f}%")

        comparison_path = os.path.join(args.output_root, "cross_dataset_comparison.json")
        with open(comparison_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Saved to {comparison_path}")


if __name__ == "__main__":
    main()