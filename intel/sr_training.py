import numpy as np
import pandas as pd
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os
import json
import argparse


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
               niterations=2000, maxsize=25, parsimony=1e-4, approach_name=None):
    y_mean = float(y_tr.mean())
    y_std = float(y_tr.std()) if y_tr.std() != 0 else 1.0
    y_scaled = (y_tr - y_mean) / y_std

    temp_filename = f"temp_pysr_{approach_name or target_name}.csv"
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
        random_state=42,
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
    args = parser.parse_args()

    all_results = {}

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

        results = run_approach_A(
            df, feature_cols, output_dir, dataset_name,
            niterations=args.niterations, maxsize=args.maxsize
        )
        all_results[dataset_name] = results["test_metrics"]

    if len(args.datasets) > 1:
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