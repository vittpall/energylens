#!/usr/bin/env python3
"""
Action-Gap Analysis — why high MAPE still yields optimal config selection
=========================================================================
`ranking_vs_mape.py` shows *empirically* that a large absolute error does not
imply a wrong optimal-configuration choice. This script states and validates
the *reason*, borrowing the action-gap argument from decision-theoretic RL.

FORMALISATION
-------------
Within one deployment scenario, config selection is a one-step decision problem
over the candidate set A (the (TP,PP) configs). Let

    t(a)  true energy of config a
    p(a)  EnergyLens prediction
    e(a)  = p(a) - t(a)                 per-config prediction error

Because argmin_a p(a) is invariant to adding a constant to p, a systematic bias
in the fit CANNOT change the selection. The decision-relevant error is therefore
the *centred* (bias-removed) error span

    eps_c = max_a | e(a) - mean_a e |          (decision-relevant error)

and NOT the MAPE, which is dominated by exactly the bias term that cancels.
Let a1, a2 be the true best and second-best configs and

    Delta = t(a2) - t(a1)                      (action gap)

THEOREM (selection certificate).  If 2*eps_c < Delta then argmin p = argmin t,
i.e. the selection is provably optimal and the regret is exactly 0.
    Proof: for any a != a1,
        p(a) - p(a1) = [t(a) - t(a1)] + [e(a) - e(a1)] >= Delta - 2*eps_c > 0,
    since the centring constant cancels in the difference e(a) - e(a1). QED

COROLLARY (regret bound).  For any prediction, regret(s) = t(argmin p) - t(a1)
is at most 2*eps_c, regardless of how large the MAPE is.

The two results predict a testable claim about the existing profiling data:

    regret is governed by the DECISION RATIO  rho = 2*eps_c / Delta,
    not by MAPE. Scenarios with rho < 1 must show zero regret; scenarios that
    lose are the near-tie ones (small Delta), not the high-MAPE ones.

WHAT THIS SCRIPT PRODUCES
-------------------------
  1. Per-scenario table of (MAPE, eps_c, Delta, rho, regret, certificate).
  2. Certificate validation: the empirical violation rate among scenarios with
     rho < 1 (the theorem says it must be 0.0%).
  3. Bound tightness: measured regret vs the 2*eps_c bound.
  4. A head-to-head discriminative comparison — ROC AUC of MAPE vs rho for
     predicting a wrong pick — quantifying that rho carries the signal and
     MAPE does not.
  5. A 6-panel figure contrasting the MAPE view against the action-gap view.

Usage:
    python action_gap.py
    python action_gap.py --n-train 50 --seeds 0 1 2
    python action_gap.py --datasets "Mistral-7B (NVIDIA A100)"
    python action_gap.py --output-dir results/action_gap
"""

import argparse, os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from compare_cross_model import (
    DATASETS, prepare_df, fit_unified, unified_predict, holdout_scenarios,
)
from ranking_vs_mape import discover_datasets
from dataset_args import resolve_datasets as resolve_dataset_args, DATASETS_HELP

C_OK, C_BAD, C_BAR, C_ALT = '#2a9d8f', '#e76f51', '#457b9d', '#8d5a97'


# ════════════════════════════════════════════════════════════
#  CORE QUANTITIES
# ════════════════════════════════════════════════════════════

def scenario_action_gap(df, true_e, pred_e):
    """Per-scenario action-gap decomposition.

    For every deployment scenario (max_tokens, batch_size) we aggregate per
    (TP,PP) config and compute the MAPE view alongside the decision-theoretic
    view (centred error span, action gap, decision ratio, regret, certificate).
    """
    work = df.copy()
    work['_true'] = true_e
    work['_pred'] = pred_e
    rows = []

    for (tok, bs), g in work.groupby(['max_tokens', 'batch_size']):
        cm = (g.groupby(['tp_degree', 'pp_degree'])
                .agg(true_e=('_true', 'mean'), pred_e=('_pred', 'mean'))
                .reset_index())
        if len(cm) < 2:
            continue
        t = cm['true_e'].values.astype(float)
        p = cm['pred_e'].values.astype(float)

        # ── absolute-error view (what the reviewer worried about) ──
        mape = float(np.mean(np.abs(p - t) / np.maximum(t, 1e-9)) * 100)

        # ── decision-theoretic view ──
        e = p - t
        bias = float(np.mean(e))                 # cancels in every pairwise diff
        eps_c = float(np.max(np.abs(e - bias)))  # decision-relevant error
        # relative version, so scenarios of different energy scale are comparable
        order = np.argsort(t)
        bi, si = int(order[0]), int(order[1])    # best, second-best (true)
        gap = float(t[si] - t[bi])               # action gap, Joules
        gap_rel = gap / max(t[bi], 1e-9) * 100

        rho = (2.0 * eps_c / gap) if gap > 1e-12 else np.inf
        certified = rho < 1.0

        # ── tight, gap-weighted certificate ──
        # The loose bound above pairs the worst-case error over ALL configs
        # against the SMALLEST gap, so a hopeless config with a large error
        # inflates it even though that config can never win. Comparing each
        # competitor's error differential to its OWN gap removes that slack:
        #     rho_t = max_{a != a1} |e(a) - e(a1)| / (t(a) - t(a1))
        # Still sufficient (the absolute value keeps it conservative), but far
        # tighter, so it certifies many more scenarios.
        others = np.arange(len(t)) != bi
        d_gap = t[others] - t[bi]
        d_err = np.abs(e[others] - e[bi])
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(d_gap > 1e-12, d_err / np.maximum(d_gap, 1e-12), np.inf)
        rho_tight = float(np.max(ratios)) if ratios.size else 0.0
        certified_tight = rho_tight < 1.0

        pi = int(np.argmin(p))                   # selected config
        regret_abs = float(t[pi] - t[bi])
        regret_pct = regret_abs / max(t[bi], 1e-9) * 100
        top1 = (pi == bi)

        rho_s = spearmanr(t, p)[0] if np.std(t) > 1e-9 and np.std(p) > 1e-9 else np.nan

        rows.append({
            'max_tokens': int(tok), 'batch_size': int(bs), 'n_configs': len(cm),
            'mape': mape, 'bias_pct': bias / max(np.mean(t), 1e-9) * 100,
            'eps_c': eps_c, 'eps_c_pct': eps_c / max(t[bi], 1e-9) * 100,
            'gap_abs': gap, 'gap_pct': gap_rel,
            'rho': rho, 'certified': certified,
            'rho_tight': rho_tight, 'certified_tight': certified_tight,
            'regret_abs': regret_abs, 'regret_pct': regret_pct,
            'bound_abs': 2.0 * eps_c, 'top1': top1, 'spearman': rho_s,
        })
    return rows


def auc_score(y_true, score):
    """ROC AUC via the rank (Mann-Whitney) identity; no sklearn dependency.

    ``y_true`` is 1 for the positive class (a wrong pick); higher ``score``
    should indicate a more likely positive.
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ════════════════════════════════════════════════════════════
#  REPORTING
# ════════════════════════════════════════════════════════════

def report_certificate(rows):
    """Validate both certificates: rho < 1 must imply zero regret."""
    top1 = np.array([r['top1'] for r in rows], dtype=bool)
    n = len(rows)

    print(f"\n{'='*72}")
    print("  1. SELECTION CERTIFICATES   (rho < 1  =>  regret 0)")
    print(f"{'='*72}")
    print(f"    Scenarios total                    : {n}")
    print(f"    Empirically correct picks          : {int(top1.sum())} "
          f"({top1.mean()*100:.1f}%)  <- ceiling any certificate can reach")

    for key, label, formula in (
        ('certified',       'LOOSE ', '2*max|e-mean(e)| / min-gap'),
        ('certified_tight', 'TIGHT ', 'max_a |e(a)-e(a1)| / (t(a)-t(a1))'),
    ):
        cert = np.array([r[key] for r in rows], dtype=bool)
        n_cert = int(cert.sum())
        print(f"\n    {label}certificate   [{formula}]")
        print(f"      Certified                        : {n_cert}  "
              f"({n_cert / n * 100:.1f}% coverage)")
        if n_cert:
            viol = int((~top1[cert]).sum())
            flag = 'OK' if viol == 0 else 'BUG — bound is exact, so this cannot happen'
            print(f"      Certificate violations           : {viol}  "
                  f"({viol / n_cert * 100:.2f}%)   [{flag}]")
            # How much of the achievable correctness does the certificate explain?
            print(f"      Share of correct picks explained : "
                  f"{n_cert / max(int(top1.sum()), 1) * 100:.1f}%")
        if (~cert).any():
            print(f"      Uncertified                      : {int((~cert).sum())}  "
                  f"of which correct anyway: {int(top1[~cert].sum())} "
                  f"({top1[~cert].mean()*100:.1f}%)")


def report_bound(rows):
    print(f"\n{'='*72}")
    print("  2. REGRET BOUND TIGHTNESS   (corollary: regret <= 2*eps_c)")
    print(f"{'='*72}")
    reg = np.array([r['regret_abs'] for r in rows])
    bnd = np.array([r['bound_abs'] for r in rows])
    ok = bnd > 1e-12
    holds = int((reg <= bnd + 1e-9).sum())
    print(f"    Bound holds on                     : {holds}/{len(rows)} scenarios "
          f"({holds/len(rows)*100:.1f}%)")
    print(f"    Mean measured regret               : {reg.mean():.3f} J")
    print(f"    Mean bound (2*eps_c)               : {bnd.mean():.3f} J")
    if ok.any():
        slack = reg[ok] / bnd[ok]
        print(f"    Mean regret/bound ratio            : {slack.mean():.3f} "
              f"(1.0 = tight, 0.0 = fully slack)")
        print(f"    Scenarios achieving zero regret    : "
              f"{int((reg <= 1e-9).sum())}/{len(rows)} "
              f"({(reg <= 1e-9).mean()*100:.1f}%)")


def report_discrimination(rows):
    """Head-to-head: does MAPE or rho predict a wrong pick?"""
    wrong = np.array([0.0 if r['top1'] else 1.0 for r in rows])
    mape = np.array([r['mape'] for r in rows])
    rho = np.array([r['rho'] for r in rows])
    gap = np.array([r['gap_pct'] for r in rows])
    regret = np.array([r['regret_pct'] for r in rows])

    print(f"\n{'='*72}")
    print("  3. WHAT ACTUALLY PREDICTS A WRONG PICK?")
    print(f"{'='*72}")
    if wrong.sum() == 0:
        print("    No wrong picks in this run — discrimination undefined.")
        print("    Re-run with a smaller --n-train to induce selection errors.")
        return
    if wrong.sum() == len(wrong):
        print("    Every pick was wrong — discrimination undefined.")
        return

    rho_t = np.array([r['rho_tight'] for r in rows])
    n_pos = int(wrong.sum())
    print(f"    Wrong picks: {n_pos}/{len(wrong)} "
          f"({n_pos/len(wrong)*100:.1f}%)"
          + ("   [WARNING: small positive class, AUCs are noisy — "
             "lower --n-train]" if n_pos < 30 else ""))

    a_mape = auc_score(wrong, mape)
    a_rho = auc_score(wrong, rho)
    a_rho_t = auc_score(wrong, rho_t)
    a_gap = auc_score(wrong, -gap)      # small gap => likely wrong
    print(f"\n    ROC AUC, predicting a WRONG pick   (0.5 = no signal)")
    print(f"      scenario MAPE                    : {a_mape:.3f}")
    print(f"      action gap (smaller = worse)     : {a_gap:.3f}")
    print(f"      loose ratio  rho = 2eps_c/gap    : {a_rho:.3f}")
    print(f"      tight ratio  rho*                : {a_rho_t:.3f}   <- the predictor")

    print(f"\n    Spearman correlation with regret%")
    for label, v in (('MAPE', mape), ('action gap %', gap),
                     ('rho (loose)', rho), ('rho* (tight)', rho_t)):
        f = np.isfinite(v)
        r = spearmanr(v[f], regret[f])[0] if f.sum() > 2 else np.nan
        print(f"      {label:<32} : {r:+.3f}")

    # Stratified control: within a narrow rho* band, does MAPE add anything?
    print(f"\n    Stratified control — within rho* bands, does MAPE still matter?")
    bands = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, np.inf)]
    any_band = False
    for lo, hi in bands:
        sel = (rho_t >= lo) & (rho_t < hi)
        if sel.sum() < 8 or len(np.unique(wrong[sel])) < 2:
            continue
        any_band = True
        a = auc_score(wrong[sel], mape[sel])
        lbl = f"rho* {lo}-{'inf' if np.isinf(hi) else hi}"
        print(f"      {lbl:<32} : n={int(sel.sum()):>4}  "
              f"n_wrong={int(wrong[sel].sum()):>3}  MAPE AUC = {a:.3f}")
    if not any_band:
        print("      (no band had enough scenarios of both classes)")
    print("      AUC ~0.5 => MAPE adds nothing once the decision ratio is known.")
    print("      AUC >>0.5 => MAPE carries residual signal the ratio misses.")


def bucket_accuracy(values, top1, edges, labels):
    idx = np.digitize(values, edges) - 1
    acc, ns = [], []
    for b in range(len(labels)):
        sel = idx == b
        acc.append(top1[sel].mean() * 100 if sel.any() else np.nan)
        ns.append(int(sel.sum()))
    return acc, ns


def _annotate_bars(ax, bars, acc, ns):
    for bar, a, n in zip(bars, acc, ns):
        if not np.isnan(a):
            ax.annotate(f'{a:.0f}%\n(n={n})',
                        xy=(bar.get_x() + bar.get_width() / 2, a),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=8)


# ════════════════════════════════════════════════════════════
#  FIGURE
# ════════════════════════════════════════════════════════════

def plot_action_gap(rows, output_dir):
    mape = np.array([r['mape'] for r in rows])
    # Panels (b) and (e) use the tight, gap-weighted ratio — the loose one
    # certifies too little to be worth a figure.
    rho = np.array([r['rho_tight'] for r in rows])
    gap = np.array([r['gap_pct'] for r in rows])
    regret = np.array([r['regret_pct'] for r in rows])
    reg_abs = np.array([r['regret_abs'] for r in rows])
    bound = np.array([r['bound_abs'] for r in rows])
    top1 = np.array([1.0 if r['top1'] else 0.0 for r in rows])
    ok = top1 == 1.0

    rho_p = np.clip(np.where(np.isfinite(rho), rho, 1e3), 1e-3, 1e3)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    (ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axes

    # (a) the OLD view — MAPE vs regret, no clean structure
    ax_a.scatter(mape[ok], regret[ok], c=C_OK, s=26, alpha=0.7, label='Correct pick')
    ax_a.scatter(mape[~ok], regret[~ok], c=C_BAD, s=26, alpha=0.85, label='Wrong pick')
    ax_a.set_xlabel('Scenario MAPE (%)')
    ax_a.set_ylabel('Selection regret (%)')
    ax_a.set_title('(a) MAPE view: no clean separation')
    ax_a.legend(fontsize=8); ax_a.grid(ls='--', alpha=0.3)

    # (b) the NEW view — decision ratio vs regret, with the certified region
    ax_b.axvspan(1e-3, 1.0, color=C_OK, alpha=0.10)
    ax_b.scatter(rho_p[ok], regret[ok], c=C_OK, s=26, alpha=0.7, label='Correct pick')
    ax_b.scatter(rho_p[~ok], regret[~ok], c=C_BAD, s=26, alpha=0.85, label='Wrong pick')
    ax_b.axvline(1.0, color='black', ls='--', lw=1.2)
    ax_b.set_xscale('log')
    ax_b.set_xlabel(r'Decision ratio  $\rho^*=\max_a|e(a)-e(a_1)|/(t(a)-t(a_1))$')
    ax_b.set_ylabel('Selection regret (%)')
    ax_b.set_title(r'(b) Action-gap view: $\rho^*<1$ certifies regret $=0$')
    ax_b.annotate('certified\noptimal', xy=(0.06, 0.72), xycoords='axes fraction',
                  fontsize=9, color=C_OK, ha='center', fontweight='bold')
    ax_b.legend(fontsize=8, loc='upper left'); ax_b.grid(ls='--', alpha=0.3)

    # (c) bound validation: measured regret vs 2*eps_c
    m = (bound > 1e-9)
    lo = max(min(bound[m].min(), max(reg_abs[reg_abs > 0].min()
                                     if (reg_abs > 0).any() else 1e-3, 1e-3)) * 0.5, 1e-4)
    hi = max(bound[m].max(), reg_abs.max()) * 2 if m.any() else 1.0
    ax_c.plot([lo, hi], [lo, hi], color='black', ls='--', lw=1.2,
              label='regret = bound (tight)')
    ax_c.scatter(bound[m], np.maximum(reg_abs[m], lo), c=C_BAR, s=26, alpha=0.7)
    ax_c.set_xscale('log'); ax_c.set_yscale('log')
    ax_c.set_xlim(lo, hi); ax_c.set_ylim(lo, hi)
    ax_c.set_xlabel(r'Bound  $2\varepsilon_c$  (J)')
    ax_c.set_ylabel('Measured regret (J)')
    ax_c.set_title('(c) Regret bound holds (all points on/below the line)')
    ax_c.legend(fontsize=8); ax_c.grid(ls='--', alpha=0.3)

    # (d) Top-1 accuracy by MAPE bucket — flat-ish
    edges_m = [0, 10, 20, 30, 50, 1e9]
    labels_m = ['0–10', '10–20', '20–30', '30–50', '50+']
    acc, ns = bucket_accuracy(mape, top1, edges_m, labels_m)
    bars = ax_d.bar(range(len(labels_m)), acc, color=C_BAR,
                    edgecolor='black', linewidth=0.5)
    _annotate_bars(ax_d, bars, acc, ns)
    ax_d.set_xticks(range(len(labels_m))); ax_d.set_xticklabels(labels_m)
    ax_d.set_xlabel('Scenario MAPE bucket (%)')
    ax_d.set_ylabel('Top-1 selection accuracy (%)')
    ax_d.set_ylim(0, 112); ax_d.axhline(100, color='gray', ls='--', alpha=0.4)
    ax_d.set_title('(d) Accuracy barely responds to MAPE')
    ax_d.grid(axis='y', ls='--', alpha=0.3)

    # (e) Top-1 accuracy by decision-ratio bucket — sharp drop past rho = 1
    edges_r = [0, 0.5, 1.0, 2.0, 5.0, 1e9]
    labels_r = ['<0.5', '0.5–1', '1–2', '2–5', '5+']
    acc, ns = bucket_accuracy(rho_p, top1, edges_r, labels_r)
    colors = [C_OK, C_OK, C_ALT, C_ALT, C_BAD]
    bars = ax_e.bar(range(len(labels_r)), acc, color=colors,
                    edgecolor='black', linewidth=0.5)
    _annotate_bars(ax_e, bars, acc, ns)
    ax_e.set_xticks(range(len(labels_r))); ax_e.set_xticklabels(labels_r)
    ax_e.axvline(1.5, color='black', ls='--', lw=1.2)
    ax_e.set_xlabel(r'Decision ratio $\rho^*$ bucket')
    ax_e.set_ylabel('Top-1 selection accuracy (%)')
    ax_e.set_ylim(0, 112); ax_e.axhline(100, color='gray', ls='--', alpha=0.4)
    ax_e.set_title(r'(e) Accuracy tracks $\rho^*$ (certified: left of line)')
    ax_e.grid(axis='y', ls='--', alpha=0.3)

    # (f) where the losses live: action-gap distribution, correct vs wrong
    gp = np.clip(gap, 1e-2, None)
    bins = np.logspace(np.log10(max(gp.min(), 1e-2)), np.log10(gp.max() + 1e-9), 22)
    ax_f.hist(gp[ok], bins=bins, color=C_OK, alpha=0.65, label='Correct pick')
    if (~ok).any():
        ax_f.hist(gp[~ok], bins=bins, color=C_BAD, alpha=0.75, label='Wrong pick')
    ax_f.set_xscale('log')
    ax_f.set_xlabel(r'Action gap $\Delta$ (% of optimal energy)')
    ax_f.set_ylabel('Scenario count')
    ax_f.set_title('(f) Losses concentrate at near-ties, not at high MAPE')
    ax_f.legend(fontsize=8); ax_f.grid(axis='y', ls='--', alpha=0.3)

    fig.suptitle('Action-gap explanation of MAPE-robust configuration selection',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(output_dir, 'action_gap.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved figure → {out}")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def resolve_datasets(args):
    if args.datasets is None:
        datasets = discover_datasets(args.results_dir, args.pattern)
        have = set(datasets.values())
        for k, v in DATASETS.items():
            if os.path.isfile(v) and v not in have:
                datasets[k] = v
        return datasets
    datasets = resolve_dataset_args(args.datasets, DATASETS, args.pattern)
    return {k: v for k, v in datasets.items() if os.path.isfile(v)}


def main():
    ap = argparse.ArgumentParser(
        description='Action-gap analysis of selection robustness under high MAPE')
    ap.add_argument('--output-dir', default='results/action_gap')
    ap.add_argument('--n-train', type=int, default=50,
                    help='Training samples for the EnergyLens fit (default 50)')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                    help='Training-sample seeds; scenarios are pooled across '
                         'seeds to get a denser picture (default 0 1 2)')
    ap.add_argument('--test-frac', type=float, default=0.3,
                    help='Fraction of scenarios (max_tokens x batch_size) held '
                         'out for scoring (default 0.3); the same split for '
                         'every seed. 0 = old in-sample behaviour')
    ap.add_argument('--split-seed', type=int, default=42,
                    help='Seed for the scenario train/test split')
    ap.add_argument('--datasets', nargs='+', default=None,
                    help=DATASETS_HELP + ' (default here: auto-discover under '
                         '--results-dir plus the DATASETS entries)')
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--pattern', default='profiling_metrics.csv')
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    datasets = resolve_datasets(args)
    if not datasets:
        print('[ERROR] No datasets available.'); return
    print(f"  Loaded {len(datasets)} dataset(s) from '{args.results_dir}'")
    print(f"  n_train={args.n_train}, seeds={args.seeds}")

    all_rows = []
    for ds_name, path in datasets.items():
        print(f"\n{'━'*72}\n  {ds_name}")
        try:
            df = prepare_df(pd.read_csv(path), path)
        except ValueError as e:
            print(f"  [SKIP] {e}"); continue

        # Hold out whole scenarios once per dataset (same split for every
        # seed); each seed draws its n_train rows from the training scenarios
        # and only the held-out scenarios are scored.
        pool_df, test_df, n_test, n_scen = holdout_scenarios(
            df, args.test_frac, args.split_seed)
        if n_test:
            print(f"  Split: {n_test}/{n_scen} scenarios held out (seed {args.split_seed})")
        elif args.test_frac > 0:
            print(f"  [split] only {n_scen} scenarios — too few to hold out; "
                  f"IN-SAMPLE (not a fair generalization test)")
        n = len(pool_df)
        true_e = test_df['measured_energy'].values.astype(float)
        ds_rows = []

        for seed in args.seeds:
            rng = np.random.RandomState(seed)
            train_df = pool_df if args.n_train >= n else \
                pool_df.iloc[rng.choice(n, args.n_train, replace=False)].reset_index(drop=True)
            try:
                params = fit_unified(train_df)
                pred_e = unified_predict(params, test_df)
            except Exception as e:
                print(f"  [SKIP seed {seed}] fit failed: {e}"); continue

            rows = scenario_action_gap(test_df, true_e, pred_e)
            for r in rows:
                r['dataset'] = ds_name
                r['seed'] = seed
            ds_rows.extend(rows)

        if not ds_rows:
            print('  [SKIP] no multi-config scenarios'); continue

        cert = np.array([r['certified'] for r in ds_rows])
        top1 = np.array([r['top1'] for r in ds_rows])
        mp = np.array([r['mape'] for r in ds_rows])
        print(f"  scenarios={len(ds_rows)}  mean MAPE={mp.mean():.1f}%  "
              f"Top-1={top1.mean()*100:.1f}%  certified={cert.mean()*100:.1f}%"
              + (f"  cert-violations={int((~top1[cert]).sum())}" if cert.any() else ''))
        all_rows.extend(ds_rows)

    if not all_rows:
        print('\n[ERROR] No scenarios computed.'); return

    report_certificate(all_rows)
    report_bound(all_rows)
    report_discrimination(all_rows)

    # ── headline sentence for the paper text ──
    mape = np.array([r['mape'] for r in all_rows])
    rho = np.array([r['rho'] for r in all_rows])
    top1 = np.array([r['top1'] for r in all_rows], dtype=float)
    rho_t = np.array([r['rho_tight'] for r in all_rows])
    hi_mape = mape >= 20.0
    hi_cert_t = hi_mape & np.array([r['certified_tight'] for r in all_rows])
    print(f"\n{'='*72}")
    print("  HEADLINE")
    print(f"{'='*72}")
    if hi_mape.any():
        print(f"    Of {int(hi_mape.sum())} scenarios with MAPE >= 20%, "
              f"{int(hi_cert_t.sum())} ({hi_cert_t.sum()/hi_mape.sum()*100:.1f}%) are")
        print(f"    PROVABLY optimal by the tight action-gap certificate, and "
              f"{top1[hi_mape].mean()*100:.1f}% select")
        print(f"    correctly in fact.")
    for lbl, v in (('loose rho', rho), ('tight rho*', rho_t)):
        fin = np.isfinite(v)
        print(f"    Median {lbl:<11} = {np.median(v[fin]):8.2f}   "
              f"(< 1 certifies zero regret)")
    print("    => Absolute accuracy is the wrong yardstick; the error must only")
    print("       be small RELATIVE TO THE ACTION GAP between candidate configs.")

    out_csv = os.path.join(args.output_dir, 'action_gap.csv')
    pd.DataFrame([{k: v for k, v in r.items()} for r in all_rows]).to_csv(
        out_csv, index=False)
    print(f"\n  Saved per-scenario CSV → {out_csv}")
    plot_action_gap(all_rows, args.output_dir)
    print(f"\n{'━'*72}\n  Done.")


if __name__ == '__main__':
    main()
