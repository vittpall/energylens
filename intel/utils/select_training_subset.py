#!/usr/bin/env python3
"""Script to select a training subset with good coverage/span of specified features.

    python select_training_subset.py --input <csv_file> --features <feat1,feat2,...> [options]
    python select_training_subset.py --input results_internVL3-9B/profiling_metrics.csv         --features tensor_parallel_size,pipeline_parallel_size,max_tokens         --output train_subset.csv --size 0.7 --method stratified
    python select_training_subset.py --input results_internVL3-9B/profiling_metrics.csv         --features tensor_parallel_size,pipeline_parallel_size         --aggregate --output train_subset.csv
"""

import argparse
import pandas as pd
import numpy as np
from itertools import product
from collections import defaultdict


def aggregate_samples_by_config(df: pd.DataFrame, config_features: list, 
                                 numeric_cols: list = None) -> pd.DataFrame:
    """
    Average multiple samples that have identical configuration values.
    
    Args:
        df: Input DataFrame with potentially multiple samples per config
        config_features: List of columns that define a unique configuration
        numeric_cols: List of numeric columns to average. If None, auto-detect.
    
    Returns:
        DataFrame with one row per unique config, numeric values averaged
    """
    # Validate config features exist
    valid_config = [f for f in config_features if f in df.columns]
    if not valid_config:
        raise ValueError(f"No valid config features found. Available: {list(df.columns)}")
    
    if len(valid_config) != len(config_features):
        missing = set(config_features) - set(valid_config)
        print(f"Warning: Config features not found: {missing}")
    
    # Auto-detect numeric columns if not specified
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove config features from numeric cols to avoid redundancy
        numeric_cols = [c for c in numeric_cols if c not in valid_config]
    
    print(f"\nAggregating samples by configuration:")
    print(f"  Config features: {valid_config}")
    print(f"  Numeric columns to average: {numeric_cols}")
    print(f"  Original samples: {len(df)}")
    
    # Count samples per config before aggregation
    samples_per_config = df.groupby(valid_config).size()
    print(f"  Unique configurations: {len(samples_per_config)}")
    print(f"  Samples per config: min={samples_per_config.min()}, max={samples_per_config.max()}, "
          f"mean={samples_per_config.mean():.1f}")
    
    # Aggregate: mean for numeric, first for others
    agg_dict = {}
    for col in df.columns:
        if col in valid_config:
            continue  # These will be in the groupby
        elif col in numeric_cols:
            agg_dict[col] = 'mean'
        else:
            agg_dict[col] = 'first'  # Keep first value for non-numeric (e.g., prompt)
    
    # Add count column to track how many samples were averaged
    df_with_count = df.copy()
    df_with_count['_sample_count'] = 1
    agg_dict['_sample_count'] = 'sum'
    
    # Perform aggregation
    df_agg = df_with_count.groupby(valid_config, as_index=False).agg(agg_dict)
    
    # Rename count column
    df_agg.rename(columns={'_sample_count': 'samples_averaged'}, inplace=True)
    
    print(f"  Aggregated samples: {len(df_agg)}")
    
    return df_agg


def analyze_feature_coverage(df: pd.DataFrame, features: list) -> dict:
    """Analyze the coverage of features in the dataset."""
    analysis = {}
    for feat in features:
        if feat not in df.columns:
            print(f"Warning: Feature '{feat}' not found in dataset")
            continue
        
        unique_vals = sorted(df[feat].unique())
        analysis[feat] = {
            'unique_values': unique_vals,
            'n_unique': len(unique_vals),
            'min': df[feat].min(),
            'max': df[feat].max(),
            'mean': df[feat].mean(),
            'std': df[feat].std(),
        }
    return analysis


def get_feature_bins(df: pd.DataFrame, feat: str, n_bins: int = 5) -> pd.Series:
    """Bin a feature into categories for stratification."""
    unique_vals = df[feat].nunique()
    
    # If few unique values, use them directly as bins
    if unique_vals <= n_bins:
        return df[feat].astype(str)
    
    # Otherwise, use quantile-based binning
    try:
        return pd.qcut(df[feat], q=n_bins, labels=False, duplicates='drop').astype(str)
    except ValueError:
        # Fall back to equal-width bins if qcut fails
        return pd.cut(df[feat], bins=n_bins, labels=False, duplicates='drop').astype(str)


def stratified_selection(df: pd.DataFrame, features: list, target_size: float = 0.7, 
                         n_bins: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Select a subset using stratified sampling across feature combinations.
    Ensures good coverage of the feature space.
    """
    np.random.seed(random_state)
    
    # Create bins for each feature
    bin_cols = []
    for feat in features:
        if feat not in df.columns:
            continue
        bin_col = f'_bin_{feat}'
        df[bin_col] = get_feature_bins(df, feat, n_bins)
        bin_cols.append(bin_col)
    
    if not bin_cols:
        raise ValueError("No valid features found for stratification")
    
    # Create combined strata
    df['_strata'] = df[bin_cols].apply(lambda x: '_'.join(x), axis=1)
    
    # Count samples per stratum
    strata_counts = df['_strata'].value_counts()
    n_strata = len(strata_counts)
    target_n = int(len(df) * target_size)
    
    print(f"\nStratification Analysis:")
    print(f"  Total samples: {len(df)}")
    print(f"  Number of strata: {n_strata}")
    print(f"  Target subset size: {target_n}")
    
    # Calculate samples per stratum (proportional, but ensure at least 1 from each)
    selected_indices = []
    
    for stratum, count in strata_counts.items():
        stratum_df = df[df['_strata'] == stratum]
        
        # Proportional allocation with minimum of 1
        n_select = max(1, int(count * target_size))
        n_select = min(n_select, count)  # Can't select more than available
        
        selected = stratum_df.sample(n=n_select, random_state=random_state)
        selected_indices.extend(selected.index.tolist())
    
    # Clean up temporary columns
    df.drop(columns=bin_cols + ['_strata'], inplace=True)
    
    subset = df.loc[selected_indices].copy()
    return subset


def coverage_greedy_selection(df: pd.DataFrame, features: list, target_size: float = 0.7,
                              random_state: int = 42) -> pd.DataFrame:
    """
    Greedy selection that maximizes feature space coverage.
    Selects samples that are most different from already selected ones.
    """
    np.random.seed(random_state)
    
    # Normalize features to [0, 1] for distance calculation
    valid_features = [f for f in features if f in df.columns]
    if not valid_features:
        raise ValueError("No valid features found")
    
    X = df[valid_features].values.astype(float)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1  # Avoid division by zero
    X_norm = (X - X_min) / X_range
    
    target_n = int(len(df) * target_size)
    n_samples = len(df)
    
    # Start with a random sample
    selected_mask = np.zeros(n_samples, dtype=bool)
    first_idx = np.random.randint(n_samples)
    selected_mask[first_idx] = True
    
    print(f"\nGreedy Coverage Selection:")
    print(f"  Target samples: {target_n}")
    
    # Greedy selection: pick the sample farthest from all selected samples
    for i in range(1, target_n):
        # Calculate minimum distance to any selected sample
        selected_points = X_norm[selected_mask]
        unselected_indices = np.where(~selected_mask)[0]
        
        if len(unselected_indices) == 0:
            break
        
        max_min_dist = -1
        best_idx = unselected_indices[0]
        
        for idx in unselected_indices:
            # Distance to closest selected point
            distances = np.linalg.norm(selected_points - X_norm[idx], axis=1)
            min_dist = distances.min()
            
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = idx
        
        selected_mask[best_idx] = True
        
        if (i + 1) % 100 == 0:
            print(f"  Selected {i + 1}/{target_n} samples...")
    
    return df.iloc[selected_mask].copy()


def unique_combinations_selection(df: pd.DataFrame, features: list, 
                                  samples_per_combo: int = 1,
                                  random_state: int = 42) -> pd.DataFrame:
    """
    Select samples ensuring all unique feature combinations are represented.
    Good for categorical/discrete features.
    """
    np.random.seed(random_state)
    
    valid_features = [f for f in features if f in df.columns]
    if not valid_features:
        raise ValueError("No valid features found")
    
    # Group by unique feature combinations
    grouped = df.groupby(valid_features)
    
    print(f"\nUnique Combinations Selection:")
    print(f"  Features: {valid_features}")
    print(f"  Unique combinations: {len(grouped)}")
    
    selected_dfs = []
    for name, group in grouped:
        n_select = min(samples_per_combo, len(group))
        selected_dfs.append(group.sample(n=n_select, random_state=random_state))
    
    subset = pd.concat(selected_dfs, ignore_index=False)
    print(f"  Selected samples: {len(subset)}")
    
    return subset


def latin_hypercube_selection(df: pd.DataFrame, features: list, target_size: float = 0.7,
                              n_bins: int = 10, random_state: int = 42) -> pd.DataFrame:
    """
    Latin Hypercube-like sampling: ensures each bin of each feature 
    is represented roughly equally.
    """
    np.random.seed(random_state)
    
    valid_features = [f for f in features if f in df.columns]
    if not valid_features:
        raise ValueError("No valid features found")
    
    target_n = int(len(df) * target_size)
    
    # Assign each sample to bins for each feature
    bin_assignments = {}
    for feat in valid_features:
        bin_assignments[feat] = get_feature_bins(df, feat, n_bins).values
    
    # Score each sample by how underrepresented its bins are
    selected_indices = []
    bin_counts = {feat: defaultdict(int) for feat in valid_features}
    
    available_indices = list(range(len(df)))
    
    print(f"\nLatin Hypercube-like Selection:")
    print(f"  Target samples: {target_n}")
    
    while len(selected_indices) < target_n and available_indices:
        # Calculate scores for remaining samples
        scores = []
        for idx in available_indices:
            # Score = sum of (1 / (count + 1)) for each bin this sample falls into
            # Lower counts = higher score = more likely to be selected
            score = 0
            for feat in valid_features:
                bin_val = bin_assignments[feat][idx]
                score += 1.0 / (bin_counts[feat][bin_val] + 1)
            scores.append(score)
        
        # Select sample with highest score (most underrepresented combination)
        scores = np.array(scores)
        # Add some randomness to avoid deterministic selection
        scores += np.random.random(len(scores)) * 0.01
        best_local_idx = np.argmax(scores)
        best_idx = available_indices[best_local_idx]
        
        # Update counts and move sample
        for feat in valid_features:
            bin_val = bin_assignments[feat][best_idx]
            bin_counts[feat][bin_val] += 1
        
        selected_indices.append(best_idx)
        available_indices.pop(best_local_idx)
        
        if len(selected_indices) % 200 == 0:
            print(f"  Selected {len(selected_indices)}/{target_n} samples...")
    
    return df.iloc[selected_indices].copy()


def evaluate_subset_coverage(df_full: pd.DataFrame, df_subset: pd.DataFrame, 
                             features: list) -> dict:
    """Evaluate how well the subset covers the feature space."""
    metrics = {}
    
    for feat in features:
        if feat not in df_full.columns:
            continue
        
        full_unique = set(df_full[feat].unique())
        subset_unique = set(df_subset[feat].unique())
        
        coverage = len(subset_unique) / len(full_unique) if full_unique else 0
        
        metrics[feat] = {
            'full_unique': len(full_unique),
            'subset_unique': len(subset_unique),
            'coverage': coverage,
            'full_range': (df_full[feat].min(), df_full[feat].max()),
            'subset_range': (df_subset[feat].min(), df_subset[feat].max()),
        }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Select a training subset with good feature space coverage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods:
  stratified  - Stratified sampling across feature bins (default)
  greedy      - Greedy selection maximizing coverage (slower but better for continuous features)
  unique      - Select from each unique feature combination
  lhs         - Latin Hypercube-like sampling

Aggregation:
  Use --aggregate to average multiple samples with identical configurations.
  Use --config-features to specify which features define a unique configuration.
  If --config-features is not specified, it defaults to the --features list.

Examples:
  python select_training_subset.py -i data.csv -f "tensor_parallel_size,pipeline_parallel_size" -o train.csv
  python select_training_subset.py -i data.csv -f "max_tokens,temperature" -m greedy -s 0.8
  
  # Aggregate 10 samples per config before selection:
  python select_training_subset.py -i data.csv -f "tensor_parallel_size,pipeline_parallel_size" --aggregate -o train.csv
  
  # Aggregate with custom config features:
  python select_training_subset.py -i data.csv -f "latency,power" --aggregate --config-features "tensor_parallel_size,pipeline_parallel_size,batch_size" -o train.csv
        """
    )
    
    parser.add_argument('-i', '--input', required=True, help='Input CSV file')
    parser.add_argument('-f', '--features', required=True, 
                        help='Comma-separated list of features to consider for selection')
    parser.add_argument('-o', '--output', default='train_subset.csv', 
                        help='Output CSV file (default: train_subset.csv)')
    parser.add_argument('-s', '--size', type=float, default=0.7,
                        help='Target subset size as fraction (default: 0.7)')
    parser.add_argument('-m', '--method', choices=['stratified', 'greedy', 'unique', 'lhs'],
                        default='stratified', help='Selection method (default: stratified)')
    parser.add_argument('--samples-per-combo', type=int, default=1,
                        help='Samples per unique combination (for "unique" method)')
    parser.add_argument('--n-bins', type=int, default=5,
                        help='Number of bins for stratification (default: 5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-remaining', action='store_true',
                        help='Also save the non-selected samples as test set')
    parser.add_argument('--aggregate', '-a', action='store_true',
                        help='Average multiple samples with identical configurations before selection')
    parser.add_argument('--config-features', '-c', 
                        help='Comma-separated list of features that define a unique configuration '
                             '(for aggregation). Defaults to --features if not specified.')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} samples")
    
    # Parse features
    features = [f.strip() for f in args.features.split(',')]
    print(f"\nSelected features for subset selection: {features}")
    
    # Parse config features (for aggregation)
    if args.config_features:
        config_features = [f.strip() for f in args.config_features.split(',')]
    else:
        config_features = features.copy()
    
    # Aggregate samples by config if requested
    if args.aggregate:
        print("\n" + "=" * 60)
        print("Aggregating samples by configuration...")
        print("=" * 60)
        df = aggregate_samples_by_config(df, config_features)
    
    # Analyze original coverage (after potential aggregation)
    print("\n" + "=" * 60)
    print("Feature Analysis" + (" (after aggregation):" if args.aggregate else ":"))
    print("=" * 60)
    analysis = analyze_feature_coverage(df, features)
    for feat, stats in analysis.items():
        print(f"\n{feat}:")
        print(f"  Unique values: {stats['n_unique']}")
        if stats['n_unique'] <= 10:
            print(f"  Values: {stats['unique_values']}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    
    # Select subset
    print("\n" + "=" * 60)
    print(f"Selecting subset using '{args.method}' method...")
    print("=" * 60)
    
    if args.method == 'stratified':
        subset = stratified_selection(df, features, args.size, args.n_bins, args.seed)
    elif args.method == 'greedy':
        subset = coverage_greedy_selection(df, features, args.size, args.seed)
    elif args.method == 'unique':
        subset = unique_combinations_selection(df, features, args.samples_per_combo, args.seed)
    elif args.method == 'lhs':
        subset = latin_hypercube_selection(df, features, args.size, args.n_bins, args.seed)
    
    # Evaluate coverage
    print("\n" + "=" * 60)
    print("Subset Coverage Evaluation:")
    print("=" * 60)
    print(f"Original size: {len(df)}")
    print(f"Subset size: {len(subset)} ({100*len(subset)/len(df):.1f}%)")
    
    metrics = evaluate_subset_coverage(df, subset, features)
    for feat, m in metrics.items():
        print(f"\n{feat}:")
        print(f"  Unique values coverage: {m['subset_unique']}/{m['full_unique']} ({100*m['coverage']:.1f}%)")
        print(f"  Original range: {m['full_range']}")
        print(f"  Subset range: {m['subset_range']}")
    
    # Save subset
    subset.to_csv(args.output, index=False)
    print(f"\nSaved training subset to: {args.output}")
    
    # Optionally save remaining as test set
    if args.save_remaining:
        remaining = df.loc[~df.index.isin(subset.index)]
        test_file = args.output.replace('.csv', '_test.csv')
        remaining.to_csv(test_file, index=False)
        print(f"Saved remaining samples (test set) to: {test_file}")


if __name__ == '__main__':
    main()
