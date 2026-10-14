#!/usr/bin/env python3
"""DoE Sampling Comparison: Random vs Latin Hypercube vs GPU-aware LHS

    python doe_sampling_comparison.py                          # all datasets, default seeds
    python doe_sampling_comparison.py --n-gpus 8              # force 8-GPU budget
    python doe_sampling_comparison.py --n-train 50 100 500
    python doe_sampling_comparison.py --n-seeds 10
    python doe_sampling_comparison.py --output-dir doe_out
    python doe_sampling_comparison.py --datasets "Mistral-7B (NVIDIA A100)"
    python doe_sampling_comparison.py --strategies random lhs sobol stratified
"""

import argparse, os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from compare_cross_model import (
    DATASETS, prepare_df,
    fit_unified, unified_predict,
    fit_maveriq, maveriq_predict,
    fit_predict_ml,
    compute_metrics,
    holdout_rows, TEST_FRAC,
    _SKLEARN,
)
from dataset_args import add_datasets_arg, resolve_datasets

try:
    from scipy.stats import qmc
    _QMC = True
except ImportError:
    _QMC = False
    print("[WARNING] scipy.stats.qmc not found – LHS/Sobol/Halton/OA-LHS strategies "
          "will fall back to random sampling.")

try:
    from sklearn.cluster import KMeans
    _SKLEARN_CLUSTER = True
except ImportError:
    _SKLEARN_CLUSTER = False
    print("[WARNING] sklearn.cluster not found – Cluster strategy will fall back to random.")

N_SEEDS_DEFAULT  = 10
N_TRAIN_DEFAULT  = [50, 100, 500]

# All strategies available in this script
SAMPLING_STRATEGIES = [
    'random',
    'lhs',
    'gpu_lhs',
    'sobol',
    'halton',
    'oa_lhs',
    'stratified',
    'cluster',
    'maxmin',
    'energy_dist',
]

# Short display labels for plots / tables
STRATEGY_LABELS = {
    'random':      'Random',
    'lhs':         'LHS',
    'gpu_lhs':     'GPU-LHS',
    'sobol':       'Sobol',
    'halton':      'Halton',
    'oa_lhs':      'OA-LHS',
    'stratified':  'Stratified',
    'cluster':     'Cluster',
    'maxmin':      'MaxMin',
    'energy_dist': 'EnergyDist',
}

# Colors and markers for up to 10 strategies
_PALETTE = [
    '#888780', '#1D9E75', '#378ADD', '#E76F51', '#9B59B6',
    '#F1C40F', '#E91E63', '#00BCD4', '#4CAF50', '#FF5722',
]
COLORS  = {s: _PALETTE[i] for i, s in enumerate(SAMPLING_STRATEGIES)}
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', 'p']
MARKERS = {s: MARKERS[i] for i, s in enumerate(SAMPLING_STRATEGIES)}


# ════════════════════════════════════════════════════════════
#  GPU-AWARE TP/PP VALIDITY
# ════════════════════════════════════════════════════════════

def infer_n_gpus(df):
    """Infer the GPU budget from the dataset: max observed tp × pp product."""
    tp = df['tp_degree'].values.astype(int)
    pp = df['pp_degree'].values.astype(int)
    return int(np.max(tp * pp))


def valid_tp_pp_pairs(n_gpus):
    """
    Return all (tp, pp) integer pairs where tp >= 1, pp >= 1, tp × pp <= n_gpus
    and both tp and pp are powers of 2 (standard practice).
    """
    powers = [2**k for k in range(0, int(np.log2(max(n_gpus, 1))) + 1)]
    return [(tp, pp) for tp in powers for pp in powers if tp * pp <= n_gpus]


# ════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ════════════════════════════════════════════════════════════

FEATURE_COLS = ['batch_size', 'max_tokens', 'tp_degree', 'pp_degree', 'total_input']


def _normalise(df):
    """Return (n_rows × 5) float array normalised to [0,1] per column."""
    X   = df[FEATURE_COLS].values.astype(float)
    lo  = X.min(axis=0)
    hi  = X.max(axis=0)
    rng = np.where(hi - lo > 0, hi - lo, 1.0)
    return (X - lo) / rng


def _nearest_unique(lhs_pts, X_norm, n_samples, rng_fallback):
    """
    Match each quasi-random point to its nearest row in X_norm (Euclidean).
    Resolve collisions greedily; fill remaining slots randomly.
    Returns a list of n_samples unique integer indices into X_norm.
    """
    from scipy.spatial.distance import cdist
    dists  = cdist(lhs_pts, X_norm)
    chosen = set()
    order  = np.argsort(dists.min(axis=1))

    for qi in order:
        row = dists[qi].copy()
        idx = int(np.argmin(row))
        while idx in chosen:
            row[idx] = np.inf
            idx      = int(np.argmin(row))
        if not np.isinf(row[idx]):
            chosen.add(idx)

    if len(chosen) < n_samples:
        pool  = list(set(range(len(X_norm))) - chosen)
        extra = rng_fallback.choice(pool,
                                    size=n_samples - len(chosen),
                                    replace=False)
        chosen.update(extra.tolist())

    return list(chosen)[:n_samples]


def _group_indices_by_tp_pp(df, n_gpus):
    """Return dict {(tp,pp): array_of_row_indices} for valid pairs present in df."""
    pairs   = valid_tp_pp_pairs(n_gpus)
    tp_arr  = df['tp_degree'].values.astype(int)
    pp_arr  = df['pp_degree'].values.astype(int)
    groups  = {}
    for tp, pp in pairs:
        mask = (tp_arr == tp) & (pp_arr == pp)
        idxs = np.where(mask)[0]
        if len(idxs) > 0:
            groups[(tp, pp)] = idxs
    return groups


def _proportional_alloc(groups, n_samples):
    """Proportional allocation of n_samples across groups by group size."""
    total_rows = sum(len(v) for v in groups.values())
    alloc = {k: max(1, round(n_samples * len(v) / total_rows))
             for k, v in groups.items()}
    while sum(alloc.values()) > n_samples:
        biggest = max(alloc, key=alloc.__getitem__)
        alloc[biggest] -= 1
    while sum(alloc.values()) < n_samples:
        biggest = max(groups, key=lambda k: len(groups[k]))
        alloc[biggest] += 1
    return alloc


# ════════════════════════════════════════════════════════════
#  SAMPLING STRATEGIES
# ════════════════════════════════════════════════════════════

# ── 1. Random ────────────────────────────────────────────────
def sample_random(df, n_samples, seed):
    """Uniform random sampling — exact replica of seed_variance baseline."""
    rng = np.random.RandomState(seed * 7 + 13)
    return rng.choice(len(df), size=n_samples, replace=False).tolist()


# ── 2. LHS ───────────────────────────────────────────────────
def sample_lhs(df, n_samples, seed):
    """
    Pure Latin Hypercube Sampling over all 5 features.
    Each factor's range is divided into n_samples equal-width strata;
    exactly one sample falls in each stratum.
    """
    if not _QMC:
        return sample_random(df, n_samples, seed)
    X_norm  = _normalise(df)
    sampler = qmc.LatinHypercube(d=X_norm.shape[1], seed=seed)
    lhs_pts = sampler.random(n=n_samples)
    rng_fb  = np.random.RandomState(seed)
    return _nearest_unique(lhs_pts, X_norm, n_samples, rng_fb)


# ── 3. GPU-aware LHS ─────────────────────────────────────────
def sample_gpu_lhs(df, n_samples, seed, n_gpus):
    """
    GPU-aware LHS: perform LHS over the 3 continuous factors
    (batch_size, max_tokens, total_input) within each valid (tp,pp) group,
    allocating samples proportionally to group sizes.
    """
    if not _QMC:
        return sample_random(df, n_samples, seed)

    groups = _group_indices_by_tp_pp(df, n_gpus)
    if not groups:
        return sample_lhs(df, n_samples, seed)

    alloc      = _proportional_alloc(groups, n_samples)
    CONT_COLS  = ['batch_size', 'max_tokens', 'total_input']
    selected   = []
    rng_fb     = np.random.RandomState(seed)

    for (tp, pp), idxs in groups.items():
        k      = alloc[(tp, pp)]
        sub_df = df.iloc[idxs].reset_index(drop=False)
        X_sub  = sub_df[CONT_COLS].values.astype(float)
        lo     = X_sub.min(axis=0); hi = X_sub.max(axis=0)
        X_norm = (X_sub - lo) / np.where(hi - lo > 0, hi - lo, 1.0)

        if k == 1 or X_sub.shape[0] == 1:
            center  = np.ones(X_norm.shape[1]) * 0.5
            nearest = int(np.argmin(np.linalg.norm(X_norm - center, axis=1)))
            selected.append(int(sub_df.iloc[nearest]['index']))
        else:
            sampler = qmc.LatinHypercube(d=X_norm.shape[1],
                                          seed=seed + tp * 100 + pp)
            pts   = sampler.random(n=k)
            local = _nearest_unique(pts, X_norm, k, rng_fb)
            selected.extend(int(sub_df.iloc[li]['index']) for li in local)

    selected = list(dict.fromkeys(selected))
    if len(selected) < n_samples:
        pool  = list(set(range(len(df))) - set(selected))
        extra = rng_fb.choice(pool,
                              size=min(n_samples - len(selected), len(pool)),
                              replace=False)
        selected.extend(extra.tolist())
    return selected[:n_samples]


# ── 4. Sobol ─────────────────────────────────────────────────
def sample_sobol(df, n_samples, seed):
    """
    Scrambled Sobol sequence (base-2 quasi-random), mapped to nearest rows.
    Sobol sequences have lower discrepancy than LHS for smooth integrands
    and scale well in higher dimensions.
    Requires n_samples to be a power of 2 for best uniformity; we use
    the next power of 2 >= n_samples and then take the first n_samples points.
    """
    if not _QMC:
        return sample_random(df, n_samples, seed)
    X_norm   = _normalise(df)
    # Sobol requires d <= 21201 and works best at power-of-2 sample counts
    n_sobol  = int(2 ** np.ceil(np.log2(max(n_samples, 2))))
    sampler  = qmc.Sobol(d=X_norm.shape[1], scramble=True, seed=seed)
    pts      = sampler.random(n_sobol)[:n_samples]
    rng_fb   = np.random.RandomState(seed)
    return _nearest_unique(pts, X_norm, n_samples, rng_fb)


# ── 5. Halton ────────────────────────────────────────────────
def sample_halton(df, n_samples, seed):
    """
    Halton low-discrepancy sequence using the first n_samples points
    with a random scramble determined by `seed`.
    Halton sequences are deterministic but their scrambled variants
    reduce correlation between dimensions.
    """
    if not _QMC:
        return sample_random(df, n_samples, seed)
    X_norm  = _normalise(df)
    sampler = qmc.Halton(d=X_norm.shape[1], scramble=True, seed=seed)
    pts     = sampler.random(n_samples)
    rng_fb  = np.random.RandomState(seed)
    return _nearest_unique(pts, X_norm, n_samples, rng_fb)


# ── 6. OA-LHS (Orthogonal Array LHS) ────────────────────────
def sample_oa_lhs(df, n_samples, seed):
    """
    Orthogonal-array-based Latin Hypercube Sampling.
    Uses scipy's LatinHypercube with strength=2, which ensures that
    all 2-dimensional projections are uniformly covered
    (i.e. no two samples share the same pair of stratum indices in any 2-D plane).
    This is a stronger space-filling guarantee than plain LHS.

    Note: strength=2 requires n_samples to be a prime power; we fall back
    to strength=1 if construction fails.
    """
    if not _QMC:
        return sample_random(df, n_samples, seed)
    X_norm = _normalise(df)
    try:
        sampler = qmc.LatinHypercube(d=X_norm.shape[1], strength=2, seed=seed)
    except (ValueError, TypeError):
        # strength=2 may fail for some (n_samples, d) combinations
        sampler = qmc.LatinHypercube(d=X_norm.shape[1], strength=1, seed=seed)
    pts    = sampler.random(n_samples)
    rng_fb = np.random.RandomState(seed)
    return _nearest_unique(pts, X_norm, n_samples, rng_fb)


# ── 7. Stratified ────────────────────────────────────────────
def sample_stratified(df, n_samples, seed, n_gpus=None):
    """
    Proportional stratified random sampling.
    Strata are defined by the unique (tp, pp) pairs present in the dataset
    (not restricted to valid GPU budget, so all observed configurations are
    eligible).  Within each stratum, samples are drawn uniformly at random.

    This guarantees that every observed (tp, pp) combination is represented
    in the training set, which is important for fitting the parallelism
    exponents γ₁ and γ₂ reliably.
    """
    rng    = np.random.RandomState(seed * 31 + 7)
    tp_arr = df['tp_degree'].values.astype(int)
    pp_arr = df['pp_degree'].values.astype(int)

    # Build groups from all observed (tp,pp) pairs
    unique_pairs = list(set(zip(tp_arr.tolist(), pp_arr.tolist())))
    groups       = {}
    for tp, pp in unique_pairs:
        mask = (tp_arr == tp) & (pp_arr == pp)
        idxs = np.where(mask)[0]
        if len(idxs) > 0:
            groups[(tp, pp)] = idxs

    alloc    = _proportional_alloc(groups, n_samples)
    selected = []
    for (tp, pp), idxs in groups.items():
        k     = alloc[(tp, pp)]
        drawn = rng.choice(idxs, size=min(k, len(idxs)), replace=False)
        selected.extend(drawn.tolist())

    selected = list(dict.fromkeys(selected))
    if len(selected) < n_samples:
        pool  = list(set(range(len(df))) - set(selected))
        extra = rng.choice(pool,
                           size=min(n_samples - len(selected), len(pool)),
                           replace=False)
        selected.extend(extra.tolist())
    return selected[:n_samples]


# ── 8. Cluster-based ─────────────────────────────────────────
def sample_cluster(df, n_samples, seed):
    """
    k-means clustering in normalised feature space.
    Partition the dataset into n_samples clusters; for each cluster draw
    one point uniformly at random from the cluster members (not necessarily
    the centroid, since the centroid may not correspond to a real data point).

    This produces a sample that is both space-filling and representative of
    the data density, unlike pure space-filling methods that may over-sample
    sparse regions.
    """
    if not _SKLEARN_CLUSTER:
        return sample_random(df, n_samples, seed)
    X_norm = _normalise(df)
    rng    = np.random.RandomState(seed)

    # Clamp k to the number of available rows
    k = min(n_samples, len(df))
    km = KMeans(n_clusters=k, random_state=seed, n_init='auto', max_iter=300)
    labels  = km.fit_predict(X_norm)
    selected = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) > 0:
            selected.append(int(rng.choice(members)))

    # If some cluster was empty (degenerate), fill with random
    selected = list(dict.fromkeys(selected))
    if len(selected) < n_samples:
        pool  = list(set(range(len(df))) - set(selected))
        extra = rng.choice(pool,
                           size=min(n_samples - len(selected), len(pool)),
                           replace=False)
        selected.extend(extra.tolist())
    return selected[:n_samples]


# ── 9. MaxMin-Distance ───────────────────────────────────────
def sample_maxmin(df, n_samples, seed):
    """
    Greedy maximin-distance (minimax optimal) sampling.
    Algorithm:
      1. Pick one point uniformly at random as the seed point.
      2. Iteratively add the point that maximises the minimum distance
         to the already-selected set (i.e. the point furthest from its
         nearest selected neighbour).
    This produces a space-filling design that maximises the minimum
    inter-sample distance, providing good coverage of sparse regions.

    Complexity: O(n_samples × N) where N is the dataset size.
    For large N this can be slow; we subsample to max 5000 candidates
    if N is very large to maintain tractability.
    """
    rng    = np.random.RandomState(seed * 17 + 3)
    X_norm = _normalise(df)
    N      = len(X_norm)

    # Subsample candidate pool for speed if dataset is large
    MAX_CAND = 5000
    if N > MAX_CAND:
        cand_idx = rng.choice(N, size=MAX_CAND, replace=False)
        X_cand   = X_norm[cand_idx]
    else:
        cand_idx = np.arange(N)
        X_cand   = X_norm

    n_select = min(n_samples, len(cand_idx))
    # Start with a random seed point
    first    = int(rng.randint(0, len(cand_idx)))
    selected = [first]
    # min_dist[i] = distance from candidate i to its nearest selected point
    min_dist = np.full(len(cand_idx), np.inf)
    min_dist -= 0  # keep as inf initially

    # Precompute distances from first point
    diff         = X_cand - X_cand[first]
    min_dist     = np.sqrt((diff ** 2).sum(axis=1))
    min_dist[first] = -1  # mark as selected

    for _ in range(n_select - 1):
        next_pt        = int(np.argmax(min_dist))
        selected.append(next_pt)
        min_dist[next_pt] = -1  # mark as selected
        # Update min distances with distances to the newly added point
        diff     = X_cand - X_cand[next_pt]
        new_dist = np.sqrt((diff ** 2).sum(axis=1))
        # Don't update already-selected points (they have distance -1)
        mask     = min_dist >= 0
        min_dist[mask] = np.minimum(min_dist[mask], new_dist[mask])

    # Map back to original dataset indices
    result = [int(cand_idx[s]) for s in selected]
    result = list(dict.fromkeys(result))
    if len(result) < n_samples:
        pool  = list(set(range(N)) - set(result))
        extra = rng.choice(pool,
                           size=min(n_samples - len(result), len(pool)),
                           replace=False)
        result.extend(extra.tolist())
    return result[:n_samples]


# ── 10. Energy-Distance ──────────────────────────────────────
def sample_energy_dist(df, n_samples, seed):
    """
    Greedy max-energy-distance sampling.
    The energy distance criterion selects the subset S that maximises
        E(S) = sum_{i in S, j in S, i≠j} ||x_i - x_j||
    i.e. the sum of all pairwise distances among selected points.
    This produces a maximally spread-out design similar to maximin but
    with a global (sum) rather than local (min) objective.

    Algorithm (greedy):
      1. Pick the point with highest average distance to all others
         (best initial spread) as the seed.
      2. Iteratively add the point that maximises the incremental increase
         in total energy: argmax_i sum_{j in S} ||x_i - x_j||.

    Complexity: O(n_samples × N). Candidate subsampling applied if N > 5000.
    """
    rng    = np.random.RandomState(seed * 23 + 11)
    X_norm = _normalise(df)
    N      = len(X_norm)

    MAX_CAND = 5000
    if N > MAX_CAND:
        cand_idx = rng.choice(N, size=MAX_CAND, replace=False)
        X_cand   = X_norm[cand_idx]
    else:
        cand_idx = np.arange(N)
        X_cand   = X_norm

    n_select = min(n_samples, len(cand_idx))
    M        = len(cand_idx)

    # Precompute full pairwise distance matrix (only feasible for M <= 5000)
    # Use float32 to halve memory
    X32    = X_cand.astype(np.float32)
    # Compute pairwise squared distances and take sqrt
    # To avoid memory spike, compute row by row if M is large
    if M <= 2000:
        diff2  = ((X32[:, None, :] - X32[None, :, :]) ** 2).sum(axis=2)
        D      = np.sqrt(diff2).astype(np.float64)
    else:
        D = np.zeros((M, M), dtype=np.float32)
        bs = 500
        for i in range(0, M, bs):
            chunk = X32[i:i+bs]
            diff2 = ((chunk[:, None, :] - X32[None, :, :]) ** 2).sum(axis=2)
            D[i:i+bs, :] = np.sqrt(diff2)
        D = D.astype(np.float64)

    # Seed: point with highest sum of distances to all others
    row_sums = D.sum(axis=1)
    first    = int(np.argmax(row_sums))
    selected = [first]
    # energy_gain[i] = sum of distances from i to all selected points so far
    energy_gain = D[first].copy()
    energy_gain[first] = -np.inf  # mark selected

    for _ in range(n_select - 1):
        next_pt = int(np.argmax(energy_gain))
        selected.append(next_pt)
        energy_gain[next_pt] = -np.inf
        # Add distances from next_pt to all candidates
        mask = energy_gain > -np.inf
        energy_gain[mask] += D[next_pt][mask]

    result = [int(cand_idx[s]) for s in selected]
    result = list(dict.fromkeys(result))
    if len(result) < n_samples:
        pool  = list(set(range(N)) - set(result))
        extra = rng.choice(pool,
                           size=min(n_samples - len(result), len(pool)),
                           replace=False)
        result.extend(extra.tolist())
    return result[:n_samples]


# ════════════════════════════════════════════════════════════
#  UNIFIED SAMPLING DISPATCHER
# ════════════════════════════════════════════════════════════

def draw_samples(df, n_samples, seed, strategy, n_gpus):
    """Dispatch to the correct sampling function."""
    n = min(n_samples, len(df))
    if strategy == 'random':
        return sample_random(df, n, seed)
    elif strategy == 'lhs':
        return sample_lhs(df, n, seed)
    elif strategy == 'gpu_lhs':
        return sample_gpu_lhs(df, n, seed, n_gpus)
    elif strategy == 'sobol':
        return sample_sobol(df, n, seed)
    elif strategy == 'halton':
        return sample_halton(df, n, seed)
    elif strategy == 'oa_lhs':
        return sample_oa_lhs(df, n, seed)
    elif strategy == 'stratified':
        return sample_stratified(df, n, seed, n_gpus)
    elif strategy == 'cluster':
        return sample_cluster(df, n, seed)
    elif strategy == 'maxmin':
        return sample_maxmin(df, n, seed)
    elif strategy == 'energy_dist':
        return sample_energy_dist(df, n, seed)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ════════════════════════════════════════════════════════════
#  EVALUATE ONE STRATEGY AT ONE n_train OVER MULTIPLE SEEDS
# ════════════════════════════════════════════════════════════

def collect_results(df, n_train, n_seeds, strategy, n_gpus, test_frac=TEST_FRAC):
    """
    Returns dict[method] = list of (mape, r2, rmse) — one entry per seed.
    'method' values: 'EnergyLens (Ours)', 'MaverIQ (Lat×P)',
                     'Linear Regression', 'Random Forest', 'Gradient Boosting'

    Per seed, a held-out test set (test_frac of the rows) is drawn from its own
    rng that depends on the seed only — never on the strategy — so every
    sampling strategy is scored on the SAME held-out rows and the strategies
    are compared like for like. The strategy then picks its n_train rows from
    the remaining pool only. test_frac=0 restores the old behaviour (pick from
    and score on the full df).
    """
    runs = {}

    # Deterministic strategies (no seed randomness in the draw itself). Their
    # picks still vary per seed because the pool they pick from does.
    DETERMINISTIC = {'maxmin', 'energy_dist'}

    for seed in range(n_seeds):
        effective_seed = 0 if strategy in DETERMINISTIC else seed

        pool_df, test_df = holdout_rows(df, np.random.RandomState(seed * 7 + 13), test_frac)
        true_e   = test_df['measured_energy'].values.astype(float)
        idx      = draw_samples(pool_df, min(n_train, len(pool_df)), effective_seed,
                                strategy, n_gpus)
        train_df = pool_df.iloc[idx].reset_index(drop=True)

        # ── EnergyLens ───────────────────────────────────────────────────
        try:
            p    = fit_unified(train_df)
            pred = unified_predict(p, test_df)
            m    = compute_metrics(pred, true_e)
        except Exception as e:
            print(f"    [WARN] EnergyLens {strategy} seed {seed}: {e}")
            m = (np.nan, np.nan, np.nan)
        runs.setdefault('EnergyLens (Ours)', []).append(m)

        # ── MaverIQ ──────────────────────────────────────────────────────
        try:
            p    = fit_maveriq(train_df)
            pred = maveriq_predict(p, test_df)
            m    = compute_metrics(pred, true_e)
        except Exception as e:
            print(f"    [WARN] MaverIQ {strategy} seed {seed}: {e}")
            m = (np.nan, np.nan, np.nan)
        runs.setdefault('MaverIQ (Lat×P)', []).append(m)

        # ── ML baselines ─────────────────────────────────────────────────
        ml = fit_predict_ml(train_df, test_df, true_e)
        for name, m in ml.items():
            runs.setdefault(name, []).append(m)

    return runs


def summarise(runs):
    summary = {}
    for method, metrics_list in runs.items():
        arr = np.array(metrics_list, dtype=float)
        summary[method] = {
            'mean': np.nanmean(arr, axis=0),
            'std':  np.nanstd(arr,  axis=0),
            'min':  np.nanmin(arr,  axis=0),
            'max':  np.nanmax(arr,  axis=0),
            'raw':  arr,
        }
    return summary


# ════════════════════════════════════════════════════════════
#  PRINT COMPARISON TABLE
# ════════════════════════════════════════════════════════════

def print_doe_table(all_results, strategies):
    model = 'EnergyLens (Ours)'
    labels = [STRATEGY_LABELS[s] for s in strategies]

    print(f"\n{'='*110}")
    print(f"  DoE COMPARISON TABLE — EnergyLens MAPE mean ± std across sampling strategies")
    print(f"  (lower MAPE and lower std both indicate a better sampling design)")
    print(f"{'='*110}")

    for ds, by_n in all_results.items():
        print(f"\n  ── {ds} ──")
        n_trains = sorted(by_n.keys())

        hdr = f"  {'n_train':<10}" + "".join(f"  {l:^26}" for l in labels)
        sub = f"  {'':<10}" + "".join(f"  {'MAPE mean ± std':^26}" for _ in labels)
        print(hdr); print(sub)
        print(f"  {'-'*95}")

        for n_train in n_trains:
            row = f"  {n_train:<10}"
            for s in strategies:
                sm = by_n[n_train].get(s, {}).get(model)
                if sm is None:
                    row += f"  {'n/a':^26}"
                else:
                    cell = f"{sm['mean'][0]:.2f}% ± {sm['std'][0]:.2f}%"
                    row += f"  {cell:^26}"
            print(row)

    print(f"\n{'='*110}")
    print(f"  FULL BREAKDOWN — All methods at n_train={min(next(iter(all_results.values())).keys())}")
    print(f"{'='*110}")
    for ds, by_n in all_results.items():
        n_train = min(by_n.keys())
        print(f"\n  ── {ds}  (n_train={n_train}) ──")
        methods = list(next(iter(next(iter(by_n.values())).values())).keys())
        hdr = f"  {'Method':<28}" + "".join(f"  {STRATEGY_LABELS[s]:^26}" for s in strategies)
        print(hdr)
        print(f"  {'-'*95}")
        for method in methods:
            row = f"  {method:<28}"
            for s in strategies:
                sm = by_n[n_train].get(s, {}).get(method)
                if sm is None:
                    row += f"  {'n/a':^26}"
                else:
                    cell = f"{sm['mean'][0]:.2f}% ± {sm['std'][0]:.2f}%"
                    row += f"  {cell:^26}"
            print(row)


# ════════════════════════════════════════════════════════════
#  PLOTS
# ════════════════════════════════════════════════════════════

def plot_strategy_comparison(all_results, output_dir, strategies):
    """Grouped bar chart comparing strategies for EnergyLens at each n_train."""
    plt.ioff()
    model  = 'EnergyLens (Ours)'
    labels = [STRATEGY_LABELS[s] for s in strategies]

    for ds, by_n in all_results.items():
        n_trains = sorted(by_n.keys())
        n_s      = len(strategies)
        x        = np.arange(len(n_trains))
        w        = min(0.8 / n_s, 0.18)
        offsets  = np.linspace(-(n_s - 1) / 2 * w, (n_s - 1) / 2 * w, n_s)

        fig, ax = plt.subplots(figsize=(max(8, n_s * 1.5), 4.5))
        for i, s in enumerate(strategies):
            means, stds = [], []
            for n_train in n_trains:
                sm = by_n[n_train].get(s, {}).get(model)
                means.append(sm['mean'][0] if sm else np.nan)
                stds.append(sm['std'][0]   if sm else 0.0)
            ax.bar(x + offsets[i], means, w,
                   yerr=stds, capsize=3,
                   label=labels[i],
                   color=COLORS[s], edgecolor='black', lw=0.5, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([f'n={n}' for n in n_trains], fontsize=10)
        ax.set_ylabel('MAPE (%)', fontsize=11)
        ax.set_title(f'Sampling Strategy Comparison — EnergyLens\n{ds}',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right', ncol=max(1, n_s // 4))
        ax.grid(axis='y', ls='--', alpha=0.35)
        fig.tight_layout()

        slug = ds.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        out  = os.path.join(output_dir, f'doe_bar_{slug}.png')
        fig.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {out}")


def plot_strategy_learning_curves(all_results, output_dir, strategies):
    """MAPE vs n_train learning curves per dataset for all strategies (EnergyLens only)."""
    plt.ioff()
    model  = 'EnergyLens (Ours)'
    labels = [STRATEGY_LABELS[s] for s in strategies]

    for ds, by_n in all_results.items():
        n_trains = sorted(by_n.keys())
        fig, ax  = plt.subplots(figsize=(7, 4.5))

        for s, label in zip(strategies, labels):
            means, stds = [], []
            for n_train in n_trains:
                sm = by_n[n_train].get(s, {}).get(model)
                means.append(sm['mean'][0] if sm else np.nan)
                stds.append(sm['std'][0]   if sm else 0.0)
            means = np.array(means); stds = np.array(stds)
            ax.plot(n_trains, means,
                    label=label, color=COLORS[s],
                    marker=MARKERS[s], lw=2, ms=6)
            ax.fill_between(n_trains, means - stds, means + stds,
                            color=COLORS[s], alpha=0.12)

        ax.set_xlabel('n_train', fontsize=11)
        ax.set_ylabel('MAPE (%) — mean ± std', fontsize=11)
        ax.set_title(f'Learning Curves — EnergyLens\n{ds}',
                     fontsize=11, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xticks(n_trains)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.legend(fontsize=8, ncol=2); ax.grid(ls='--', alpha=0.35)
        fig.tight_layout()

        slug = ds.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        out  = os.path.join(output_dir, f'doe_curve_{slug}.png')
        fig.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {out}")


def plot_variance_reduction_heatmap(all_results, output_dir, strategies):
    """Heatmap: MAPE std at smallest n_train for EnergyLens across strategies and datasets."""
    plt.ioff()
    datasets = list(all_results.keys())
    model    = 'EnergyLens (Ours)'
    n_min    = min(min(by_n.keys()) for by_n in all_results.values())
    labels   = [STRATEGY_LABELS[s] for s in strategies]

    data = np.full((len(datasets), len(strategies)), np.nan)
    for ri, ds in enumerate(datasets):
        for ci, s in enumerate(strategies):
            sm = all_results[ds].get(n_min, {}).get(s, {}).get(model)
            if sm is not None:
                data[ri, ci] = sm['std'][0]

    fig, ax = plt.subplots(figsize=(max(6, len(strategies) * 0.9),
                                    max(4, len(datasets) * 0.7)))
    vmax = np.nanpercentile(data, 95) if not np.all(np.isnan(data)) else 1.0
    im   = ax.imshow(data, cmap='YlOrRd', vmin=0, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax, label=f'MAPE std (%) at n_train={n_min}')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha='right')
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=9)
    for ri in range(len(datasets)):
        for ci in range(len(strategies)):
            v = data[ri, ci]
            if not np.isnan(v):
                tc = 'white' if v / max(vmax, 1e-9) > 0.6 else 'black'
                ax.text(ci, ri, f'{v:.1f}%', ha='center', va='center',
                        fontsize=8, color=tc)
    ax.set_title(f'MAPE Std — EnergyLens at n_train={n_min}\n'
                 f'(lower = more reliable sampling design)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(output_dir, 'doe_variance_heatmap.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_ranking_summary(all_results, output_dir, strategies):
    """
    For each dataset and n_train, rank strategies by EnergyLens MAPE mean.
    Plot a heatmap of ranks (1 = best).
    """
    plt.ioff()
    model    = 'EnergyLens (Ours)'
    datasets = list(all_results.keys())
    n_trains = sorted(set(n for by_n in all_results.values() for n in by_n))
    labels   = [STRATEGY_LABELS[s] for s in strategies]

    # rows = (dataset × n_train) combinations, cols = strategies
    row_labels = []
    data       = []
    for ds in datasets:
        for n_train in n_trains:
            if n_train not in all_results[ds]:
                continue
            row_labels.append(f'{ds[:18]}… n={n_train}' if len(ds) > 18
                               else f'{ds} n={n_train}')
            means = []
            for s in strategies:
                sm = all_results[ds][n_train].get(s, {}).get(model)
                means.append(sm['mean'][0] if sm else np.nan)
            # Rank: 1 = best (lowest MAPE), nan = unranked
            arr   = np.array(means, dtype=float)
            valid = ~np.isnan(arr)
            ranks = np.full(len(arr), np.nan)
            if valid.sum() > 0:
                order = np.argsort(arr[valid])
                ranked_idx = np.where(valid)[0][order]
                for r, idx in enumerate(ranked_idx):
                    ranks[idx] = r + 1
            data.append(ranks)

    data = np.array(data)
    if data.size == 0:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(strategies) * 0.9),
                                    max(4, len(row_labels) * 0.45)))
    im = ax.imshow(data, cmap='RdYlGn_r', vmin=1, vmax=len(strategies),
                   aspect='auto')
    plt.colorbar(im, ax=ax, label='Rank (1 = best MAPE)')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for ri in range(len(row_labels)):
        for ci in range(len(strategies)):
            v = data[ri, ci]
            if not np.isnan(v):
                tc = 'white' if v > len(strategies) * 0.6 else 'black'
                ax.text(ci, ri, f'{int(v)}', ha='center', va='center',
                        fontsize=8, color=tc)
    ax.set_title('Strategy Rankings — EnergyLens MAPE\n(1 = lowest MAPE = best)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(output_dir, 'doe_ranking_heatmap.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_gpu_coverage(df, n_gpus, output_dir, ds_slug):
    """Scatter: observed (tp,pp) pairs vs valid GPU-budget pairs."""
    plt.ioff()
    pairs_valid   = set(valid_tp_pp_pairs(n_gpus))
    tp_arr        = df['tp_degree'].values.astype(int)
    pp_arr        = df['pp_degree'].values.astype(int)
    pairs_in_data = set(zip(tp_arr.tolist(), pp_arr.tolist()))
    only_valid    = pairs_valid - pairs_in_data
    in_data_only  = pairs_in_data - pairs_valid
    both          = pairs_valid & pairs_in_data

    fig, ax = plt.subplots(figsize=(5, 4))
    if only_valid:
        tps, pps = zip(*only_valid)
        ax.scatter(tps, pps, marker='x', s=60, color='#B4B2A9', lw=1.5,
                   label='valid, not in data')
    if in_data_only:
        tps, pps = zip(*in_data_only)
        ax.scatter(tps, pps, marker='D', s=60, color='#F09595',
                   label='in data, exceeds budget')
    if both:
        tps, pps = zip(*both)
        ax.scatter(tps, pps, marker='o', s=80, color='#1D9E75',
                   label='valid & in data')

    ax.set_xlabel('TP degree', fontsize=11)
    ax.set_ylabel('PP degree', fontsize=11)
    ax.set_title(f'GPU coverage — n_gpus={n_gpus}\n{ds_slug}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(ls='--', alpha=0.35)
    fig.tight_layout()
    out = os.path.join(output_dir, f'gpu_coverage_{ds_slug}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out}")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='DoE sampling comparison: Random vs LHS vs GPU-aware LHS '
                    'vs Sobol vs Halton vs OA-LHS vs Stratified vs Cluster '
                    'vs MaxMin vs EnergyDist')
    ap.add_argument('--output-dir', default='doe_comparison')
    ap.add_argument('--n-train',    type=int, nargs='+', default=N_TRAIN_DEFAULT)
    ap.add_argument('--n-seeds',    type=int, default=N_SEEDS_DEFAULT,
                    help='Random seeds per strategy (default: 10)')
    ap.add_argument('--n-gpus',     type=int, default=None,
                    help='GPU count for tp×pp budget (default: inferred from data)')
    ap.add_argument('--no-plots',   action='store_true')
    ap.add_argument('--test-frac',  type=float, default=TEST_FRAC,
                    help='Fraction of rows held out per seed for scoring; the '
                         'same rows for every strategy (default 0.2). '
                         '0 = pick from and score on the full dataset')
    add_datasets_arg(ap)
    ap.add_argument('--strategies', nargs='+', default=SAMPLING_STRATEGIES,
                    choices=SAMPLING_STRATEGIES,
                    help=f'Subset of strategies (default: all). '
                         f'Choices: {SAMPLING_STRATEGIES}')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not _QMC and any(s in args.strategies
                        for s in ['lhs', 'gpu_lhs', 'sobol', 'halton', 'oa_lhs']):
        print("[WARNING] scipy.stats.qmc unavailable — quasi-random strategies "
              "will fall back to random sampling.")
    if not _SKLEARN_CLUSTER and 'cluster' in args.strategies:
        print("[WARNING] sklearn.cluster unavailable — Cluster strategy "
              "will fall back to random sampling.")

    datasets = resolve_datasets(args.datasets, DATASETS)

    missing  = [(n, p) for n, p in datasets.items() if not os.path.isfile(p)]
    if missing:
        print('[WARNING] Missing datasets:')
        for n, p in missing:
            print(f'  {n}: {p}')
        datasets = {k: v for k, v in datasets.items() if (k, v) not in missing}
    if not datasets:
        print('[ERROR] No datasets available.'); return

    all_results = {}
    csv_rows    = []

    for ds_name, ds_path in datasets.items():
        print(f"\n{'━'*70}")
        print(f"  Dataset : {ds_name}")
        raw = pd.read_csv(ds_path)
        print(f"  Rows    : {len(raw)}")
        try:
            df = prepare_df(raw, ds_path)
        except ValueError as e:
            print(f"  [SKIP] {e}"); continue

        n_gpus = args.n_gpus if args.n_gpus else infer_n_gpus(df)
        print(f"  GPU budget (tp×pp): {n_gpus}")
        print(f"  Valid (tp,pp) pairs: {valid_tp_pp_pairs(n_gpus)}")

        all_results[ds_name] = {}

        for n_train in sorted(args.n_train):
            pool_n = len(df) - (max(1, int(round(args.test_frac * len(df))))
                                if args.test_frac > 0 else 0)
            if n_train > pool_n:
                print(f"  [SKIP] n_train={n_train} > training pool {pool_n} "
                      f"(dataset {len(df)} minus held-out test rows)")
                continue
            all_results[ds_name][n_train] = {}
            print(f"\n  ── n_train={n_train} ──")

            for strategy in args.strategies:
                print(f"    strategy={strategy}  ({args.n_seeds} seeds)")
                runs    = collect_results(df, n_train, args.n_seeds,
                                          strategy, n_gpus, args.test_frac)
                summary = summarise(runs)
                all_results[ds_name][n_train][strategy] = summary

                for method, s in summary.items():
                    print(f"      {method:<28}  "
                          f"MAPE {s['mean'][0]:6.2f}% ± {s['std'][0]:5.2f}%  "
                          f"R² {s['mean'][1]:.4f} ± {s['std'][1]:.4f}")
                    csv_rows.append({
                        'dataset':    ds_name,
                        'n_train':    n_train,
                        'strategy':   strategy,
                        'n_gpus':     n_gpus,
                        'method':     method,
                        'mape_mean':  round(s['mean'][0], 3),
                        'mape_std':   round(s['std'][0],  3),
                        'mape_min':   round(s['min'][0],  3),
                        'mape_max':   round(s['max'][0],  3),
                        'r2_mean':    round(s['mean'][1], 4),
                        'r2_std':     round(s['std'][1],  4),
                        'rmse_mean':  round(s['mean'][2], 4),
                        'rmse_std':   round(s['std'][2],  4),
                        'n_seeds':    args.n_seeds,
                    })

        if not args.no_plots:
            slug = (ds_name.replace('/', '_').replace(' ', '_')
                           .replace('(', '').replace(')', ''))
            plot_gpu_coverage(df, n_gpus, args.output_dir, slug)

    if csv_rows:
        csv_path = os.path.join(args.output_dir, 'doe_results.csv')
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"\n  Saved CSV → {csv_path}")

    print_doe_table(all_results, args.strategies)

    if not args.no_plots and all_results:
        print(f"\n  Generating plots → {args.output_dir}/")
        plot_strategy_comparison(all_results, args.output_dir, args.strategies)
        plot_strategy_learning_curves(all_results, args.output_dir, args.strategies)
        plot_variance_reduction_heatmap(all_results, args.output_dir, args.strategies)
        plot_ranking_summary(all_results, args.output_dir, args.strategies)

    print(f"\n{'━'*70}")
    print("  Done.")


if __name__ == '__main__':
    main()