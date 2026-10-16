#!/usr/bin/env python3
"""Term-by-term ablation of the EnergyLens template, and a latency x power physics model.

Every variant is fitted on the same 50 random rows of the training scenarios and scored on
the held-out scenarios (MAPE, pairwise accuracy, Top-1, regret), over the 29 A100 datasets
and three split seeds:
  full                 Eq. (unified-tp-pp)
  no overhead          delta_1 = delta_2 = 0
  symmetric exponent   one parallelism exponent per phase (gamma_1 = gamma_2)
  linear batch         uncapped batch size, beta = 1, epsilon = 0 (energy per request ~ 1/b)
  latency x power      log-linear latency model x linear power model, divided by the batch
  SR raw formula       SR's best formula of the dataset (seed 0), all constants re-fitted
                       (only the five datasets SR was run on)

    python test/ablation.py [--seeds 42 7 123] [--workers 16]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NVIDIA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(NVIDIA, "..", "adaptive_sampling"))
from simulator import PATHS, load_csv, ranking_metrics  # noqa: E402
from compare_cross_model import holdout_scenarios, effective_batch, fit_unified, unified_predict  # noqa: E402

SR = {  # dataset label in PATHS -> SR result folder
    "Mistral-7B": "run_text_only_Mistral-7B-Instruct-v0.3",
    "Nemotron-Nano-9B": "run_text_only_NVIDIA-Nemotron-Nano-9B-v2",
    "Qwen2-VL-7B": "run_text_only_Qwen2-VL-7B-Instruct",
    "Qwen2-VL-7B (image)": "run_text_image_Qwen2-VL-7B-Instruct",
    "LLaVA-1.5-7B": "run_text_only_llava-1.5-7b-hf",
}
BOUNDS = [(1e-8, 1e6), (0.01, 3), (-2, 3), (-2, 3), (0, 100),
          (1e-8, 1e6), (0.01, 3), (-2, 3), (-2, 3), (0, 100),
          (-1e6, 1e6), (-1e6, 1e6)]


def template(p, d, bs):
    a_p, b_p, g1p, g2p, e_p, a_d, b_d, g1d, g2d, e_d, d1, d2 = p
    tok, tp, pp = d["decode_tokens"].values.astype(float), d["tp_degree"].values.astype(float), d["pp_degree"].values.astype(float)
    inp = d["total_input"].values.astype(float)
    return (a_p * inp / (bs ** b_p + e_p) * tp ** g1p * pp ** g2p
            + a_d * tok / (bs ** b_d + e_d) * tp ** g1d * pp ** g2d + d1 * tp + d2 * pp)


def fit_variant(train, expand, n_free, x0_full, restarts=10, linear_batch=False):
    """Fit a constrained template: expand(free) -> 12 parameters."""
    y = train["measured_energy"].values.astype(float)
    bs = train["batch_size"].values.astype(float) if linear_batch else effective_batch(train)

    def loss(q):
        return np.mean(np.abs(template(expand(q), train, bs) - y) / y)
    best, bq = np.inf, None
    for i in range(restarts):
        rng = np.random.RandomState(i * 42)
        x0 = x0_full(rng)
        r = minimize(loss, x0, method="L-BFGS-B", bounds=free_bounds(n_free, expand.__name__),
                     options={"maxiter": 5000})
        if r.fun < best:
            best, bq = r.fun, r.x
    p = expand(bq)
    return lambda d: template(p, d, d["batch_size"].values.astype(float) if linear_batch else effective_batch(d))


def free_bounds(n, name):
    if name == "sym":      # a_p b_p g_p e_p a_d b_d g_d e_d d1 d2
        return [BOUNDS[i] for i in (0, 1, 2, 4, 5, 6, 7, 9, 10, 11)]
    if name == "nodelta":  # first ten
        return BOUNDS[:10]
    if name == "linb":     # a_p g1p g2p a_d g1d g2d d1 d2
        return [BOUNDS[i] for i in (0, 2, 3, 5, 7, 8, 10, 11)]
    raise ValueError(name)


def starts(train):
    y = train["measured_energy"].values.astype(float)
    med = np.median(y)
    sp = med * 0.3 / max(np.median(train["total_input"].values), 1)
    sd = med * 0.7 / max(np.median(train["decode_tokens"].values), 1)

    def full(rng):
        return [sp * rng.uniform(.1, 3), rng.uniform(.3, 1.5), rng.uniform(-1.2, -.2), rng.uniform(-1, 0), rng.uniform(.01, 2),
                sd * rng.uniform(.1, 3), rng.uniform(.3, 1.5), rng.uniform(-1.2, -.2), rng.uniform(-1, 0), rng.uniform(.01, 2),
                rng.uniform(-med * .05, med * .1), rng.uniform(-med * .05, med * .1)]
    return full


def variants(train):
    full = starts(train)
    out = {}
    th = fit_unified(train, restarts=10)
    out["full"] = lambda d: unified_predict(th, d)

    def nodelta(q):
        return list(q) + [0.0, 0.0]
    out["no overhead"] = fit_variant(train, nodelta, 10, lambda r: full(r)[:10])

    def sym(q):
        a_p, b_p, g_p, e_p, a_d, b_d, g_d, e_d, d1, d2 = q
        return [a_p, b_p, g_p, g_p, e_p, a_d, b_d, g_d, g_d, e_d, d1, d2]
    out["symmetric exponent"] = fit_variant(train, sym, 10, lambda r: [full(r)[i] for i in (0, 1, 2, 4, 5, 6, 7, 9, 10, 11)])

    def linb(q):
        a_p, g1p, g2p, a_d, g1d, g2d, d1, d2 = q
        return [a_p, 1.0, g1p, g2p, 0.0, a_d, 1.0, g1d, g2d, 0.0, d1, d2]
    out["linear batch"] = fit_variant(train, linb, 8, lambda r: [full(r)[i] for i in (0, 2, 3, 5, 7, 8, 10, 11)],
                                      linear_batch=True)
    out["latency x power"] = physics(train)
    return out


def physics(train):
    """log L = linear in (log tp, log pp, log b, log T_out, log T_in); P = linear in (n, tp, log b)."""
    def X(d):
        tp, pp = d["tp_degree"].values.astype(float), d["pp_degree"].values.astype(float)
        b = d["batch_size"].values.astype(float)
        return np.column_stack([np.ones(len(d)), np.log(tp), np.log(pp), np.log(b),
                                np.log(d["decode_tokens"].values.astype(float) + 1),
                                np.log(d["total_input"].values.astype(float) + 1)]), tp, pp, b
    Xl, tp, pp, b = X(train)
    wl = np.linalg.lstsq(Xl, np.log(train["measured_latency"].values.astype(float)), rcond=None)[0]
    Xp = np.column_stack([np.ones(len(train)), tp * pp, tp, np.log(b)])
    wp = np.linalg.lstsq(Xp, train["gpu_power_w"].values.astype(float), rcond=None)[0]

    def pred(d):
        Xd, t, p, bb = X(d)
        lat = np.exp(Xd @ wl)
        pw = np.column_stack([np.ones(len(d)), t * p, t, np.log(bb)]) @ wp
        return np.maximum(pw, 1.0) * lat / bb       # energy per request = power x latency / batch
    return pred


def sr_raw(name, train):
    import sympy as sp
    path = os.path.join(NVIDIA, "sr_results", SR[name])
    eq = pd.read_csv([os.path.join(path, f) for f in os.listdir(path) if f.startswith("seed_equations")][0])
    col = "best_sympy" if "best_sympy" in eq else "sympy"
    expr = sp.sympify(eq.sort_values("seed").iloc[0][col])
    nums = sorted({a for a in expr.atoms(sp.Number) if a not in (0, 1, -1)}, key=str)
    cs = sp.symbols(f"c0:{len(nums)}")
    expr_c = expr.xreplace(dict(zip(nums, cs)))
    syms = sorted(expr_c.free_symbols - set(cs), key=str)
    f = sp.lambdify(list(cs) + syms, expr_c, "numpy")

    def env(d):
        tp, pp = d["tp_degree"].values.astype(float), d["pp_degree"].values.astype(float)
        base = {"max_tokens": d["max_tokens"].values.astype(float), "batch_size": d["batch_size"].values.astype(float),
                "tensor_parallel_size": tp, "pipeline_parallel_size": pp, "parallelism": tp * pp,
                "input_length": d["input_length"].values.astype(float) if "input_length" in d else d["total_input"].values / d["batch_size"].values}
        base["total_input_tokens"] = base["input_length"] * base["batch_size"]
        base["tokens_per_gpu"] = base["total_input_tokens"] / (tp * pp)
        base["workload_per_gpu"] = base["tokens_per_gpu"]
        return [base[str(s)] for s in syms]
    y = train["measured_energy"].values.astype(float)
    E = env(train)
    c0 = np.array([float(n) for n in nums])

    def model(q, e):
        return q[0] * np.asarray(f(*q[2:], *e), float) + q[1]

    def loss(q):
        r = model(q, E)
        return np.mean(np.abs(r - y) / y) if np.all(np.isfinite(r)) else 1e9
    best, bq = np.inf, None
    for i in range(10):
        rng = np.random.RandomState(i)
        x0 = np.concatenate([[np.std(y) * rng.uniform(.5, 2), np.mean(y)], c0 * rng.uniform(.5, 1.5, len(c0))])
        r = minimize(loss, x0, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10})
        if r.fun < best:
            best, bq = r.fun, r.x
    return lambda d: model(bq, env(d)), str(expr)


def score(pred, test):
    y = test["measured_energy"].values.astype(float)
    p = np.asarray(pred, float)
    return {"mape": 100 * np.mean(np.abs(p - y) / y), **ranking_metrics(test, p)}


def run_one(task):
    name, seed = task
    rows = []
    try:
        d = load_csv(PATHS[name])
        pool, test, _, _ = holdout_scenarios(d, 0.3, seed)
        train = pool.sample(min(50, len(pool)), random_state=seed)
        for v, f in variants(train).items():
            rows.append({"dataset": name, "seed": seed, "variant": v, **score(f(test), test)})
        if name in SR:
            try:
                f, expr = sr_raw(name, train)
                rows.append({"dataset": name, "seed": seed, "variant": "SR raw formula", **score(f(test), test), "expr": expr})
            except Exception as e:                       # SR output in another format: keep the other variants
                print(f"[no SR] {name} {seed}: {type(e).__name__}: {e}", flush=True)
    except Exception as e:
        print(f"[skip] {name} {seed}: {type(e).__name__}: {e}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--workers", type=int, default=int(os.environ.get("NCPUS", 16)))
    ap.add_argument("--out", default=os.path.join(NVIDIA, "results_figures", "ablation"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = [(n, s) for n in PATHS for s in a.seeds]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(run_one, tasks):
            rows += r
            if r:
                print(f"done {r[0]['dataset']} seed {r[0]['seed']}", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(a.out, "ablation_rows.csv"), index=False)
    cols = ["mape", "pairwise", "top1", "regret"]
    per = r.groupby(["variant", "dataset"])[cols].mean()
    s = per.groupby("variant").agg(["mean"]).droplevel(1, axis=1)
    s["datasets"] = per.groupby("variant").size()
    s.round(2).to_csv(os.path.join(a.out, "ablation_summary.csv"))
    pd.set_option("display.width", 200)
    print(s.round(2).to_string())
    sr = r[r.dataset.isin(SR)]
    print("\non the SR datasets only:")
    print(sr.groupby("variant")[cols].mean().round(2).to_string())


if __name__ == "__main__":
    main()
