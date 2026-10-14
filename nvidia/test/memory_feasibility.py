#!/usr/bin/env python3
"""Predict vLLM memory per configuration, and which configurations fit.

Before predicting energy one must know which (TP, PP, token budget) engines start
at all, and how many tokens of KV cache they get (which caps the requests that run
at once). MaverIQ estimates memory from three reference layouts per model,
    M(TP, PP) = M(1,1) + (TP-1) A + TP (PP-1) A'           (memory_estimator.py)
fitted on measured total memory. vLLM instead reserves a fixed share of every GPU
(gpu_memory_utilization = 0.9) and fills what the weights and the activation peak
leave with KV cache, so what decides feasibility is the busiest GPU's weights and
its KV capacity. We compare three predictors of those, scored against the engine
starts in the profiling logs (test/memory_logs.py):

  file-based      weights per GPU from the checkpoint's tensor shapes (safetensors
                  headers), split over pipeline stages like vLLM (get_pp_indices)
                  and divided by TP; activation/runtime overhead from a 3-term
                  linear model fitted on the OTHER models (leave-one-model-out), so
                  the target model is never run.
  MaverIQ-form    the 3-reference form above on weights and on non-KV memory per
                  model, fitted on its (1,1), (1,2), (2,1) engines at the same budget;
                  undefined when (1,1) does not fit (MaverIQ then uses fingerprints).
  naive           parameters x 2 bytes / (TP x PP), no overhead.

    python test/memory_feasibility.py      # -> results_figures/memory/
"""
import glob
import json
import os
import re
import struct
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NVIDIA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from memory_logs import collect  # noqa: E402

OUT = os.path.join(NVIDIA, "results_figures", "memory")
MODEL_ROOTS = [p for p in (os.environ.get("HF_HUB_CACHE"), os.environ.get("MODEL_DIR")) if p]
GIB = 2 ** 30
GPU_TOTAL_GIB = 39.394          # A100-SXM4-40GB as torch reports it (42,298,834,944 B)
UTIL = 0.9                      # gpu_memory_utilization (source/model.py)
BUDGET_GIB = UTIL * GPU_TOTAL_GIB
BLOCK = 16                      # vLLM KV block size (tokens)
ATTN_ONLY = {"mamba", "falcon_mamba", "nemotron_h"}   # model_type with state caches: no token KV check

LAYER_RX = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)\.")
VISION_RX = re.compile(r"vision|visual|mlp1|multi_modal_projector|image_newline|vpm")
EMBED_RX = re.compile(r"embed_tokens|embeddings\.weight$|wte|tok_embeddings")
HEAD_RX = re.compile(r"lm_head|(?:^|\.)norm(?:_f)?\.weight$|final_layernorm")


# ── checkpoint inventory ─────────────────────────────────────────────────────
def snapshot(model_id):
    name = "models--" + model_id.replace("/", "--")
    for root in MODEL_ROOTS:
        for d in sorted(glob.glob(os.path.join(root, name, "snapshots", "*"))):
            if glob.glob(os.path.join(d, "*.safetensors")):
                return d
    return None


def config(model_id):
    """config.json from any cached copy (some hub snapshots hold only the weights)."""
    name = "models--" + model_id.replace("/", "--")
    for root in MODEL_ROOTS:
        for p in sorted(glob.glob(os.path.join(root, name, "snapshots", "*", "config.json"))):
            return json.load(open(p))
    return {}


def tensors(model_id):
    """[(name, n_elements)] read from the safetensors headers (no weights loaded)."""
    out = []
    files = [f for f in sorted(glob.glob(os.path.join(snapshot(model_id), "*.safetensors")))
             if not os.path.basename(f).startswith("consolidated")]   # Mistral's duplicate copy
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                out.append((k, int(np.prod(v["shape"])) if v["shape"] else 1, len(v["shape"])))
    return out


def text_config(model_id, cfg):
    """The language model's config with library defaults filled in (llava-hf's
    text_config only stores the fields that differ from Llama's defaults)."""
    from transformers import AutoConfig
    try:
        full = AutoConfig.for_model(**cfg).to_dict()
    except Exception:          # remote-code models (InternVL): the raw sub-config is complete
        full = cfg
    for k in ("text_config", "llm_config", "language_config"):
        if isinstance(full.get(k), dict) and full[k].get("num_hidden_layers"):
            return full[k]
    return full


def pp_partitions(n_layers, pp):
    parts = [n_layers // pp] * pp
    for i in range(2, n_layers % pp + 2):
        parts[-i] += 1
    return parts


class Inventory:
    """Per-stage weight bytes for any (TP, PP), from the checkpoint."""

    def __init__(self, model_id):
        self.model_id = model_id
        self.cfg = config(model_id)
        tc = text_config(model_id, self.cfg)
        self.n_layers = tc["num_hidden_layers"]
        self.tied = bool(self.cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings", False)))
        self.model_type = tc.get("model_type", self.cfg.get("model_type", ""))
        self.hidden = tc.get("hidden_size") or tc.get("d_model")
        self.vocab = tc.get("vocab_size")
        self.inter = tc.get("intermediate_size") or 4 * self.hidden
        heads = tc.get("num_attention_heads") or 1
        self.kv_heads = tc.get("num_key_value_heads") or heads
        self.head_dim = tc.get("head_dim") or self.hidden // heads
        self.layer = np.zeros(self.n_layers)      # params per decoder layer
        self.layer_lin = np.zeros(self.n_layers)  # of which 2-D (quantisable by bnb)
        self.first = self.last = self.embed = 0.0
        for name, n, ndim in tensors(model_id):
            m = LAYER_RX.search(name)
            if VISION_RX.search(name):
                self.first += n
            elif m and int(m[1]) < self.n_layers and not VISION_RX.search(name):
                self.layer[int(m[1])] += n
                self.layer_lin[int(m[1])] += n if ndim == 2 else 0
            elif EMBED_RX.search(name):
                self.first += n
                self.embed += n
            elif HEAD_RX.search(name):
                self.last += n
            else:
                self.first += n
        if self.tied:
            self.last += self.embed               # vLLM keeps a tied lm_head on the last stage
        self.params = self.layer.sum() + self.first + self.last - (self.embed if self.tied else 0)

    def stage_gib(self, tp, pp, quant="none"):
        """Weight GiB on each pipeline stage's TP rank 0 (vLLM loads bf16; bnb: 4-bit linears)."""
        out, start = [], 0
        for s, k in enumerate(pp_partitions(self.n_layers, pp)):
            lin = self.layer_lin[start:start + k].sum()
            rest = self.layer[start:start + k].sum() - lin
            b = (lin * (0.5 + 4 / 64) if quant == "bitsandbytes" else lin * 2) + rest * 2
            if s == 0:
                b += self.first * 2
            if s == pp - 1:
                b += self.last * 2 if (pp > 1 or not self.tied) else 0
            out.append(b / tp / GIB)
            start += k
        return out

    def kv_bytes_per_token(self, tp, pp):
        """Per GPU, for the stage with the most layers (vLLM sizes every stage alike)."""
        layers = max(pp_partitions(self.n_layers, pp))
        return 2 * layers * max(self.kv_heads // tp, 1) * self.head_dim * 2


# ── data ─────────────────────────────────────────────────────────────────────
def engines():
    """One row per (model, quantization, TP, PP, budget): medians of the finished
    starts, and fits = any start finished; configs whose starts all died for a
    reason other than memory (walltime, kill) are dropped."""
    e = collect()
    e = e[e["ok"] | (e["error"] != "")]
    keys = ["model", "quantization", "tp", "pp", "budget", "max_seq_len"]
    g = e.groupby(keys)
    out = g.agg(fits=("ok", "any"), weights_gib=("weights_gib", "median"),
                kv_avail_gib=("kv_avail_gib", "median"), kv_tokens=("kv_tokens", "median"),
                starts=("ok", "size")).reset_index()
    out["overhead_gib"] = BUDGET_GIB - out["weights_gib"] - out["kv_avail_gib"]
    return out


# ── predictors ───────────────────────────────────────────────────────────────
def overhead_features(inv, tp, pp, budget):
    """Activation peak of vLLM's profiling run on stage 0: a constant, the MLP/attention
    activations of `budget` tokens (sharded by TP), and the vocab-sized logits of the
    sampler (on the last stage, so on stage 0 only when PP = 1)."""
    return np.array([1.0,
                     budget * (inv.hidden + inv.inter / tp) * 2 / GIB,
                     (pp == 1) * budget * inv.vocab / tp * 4 / GIB])


def fit_overhead(rows, invs):
    X = np.array([overhead_features(invs[r.model], r.tp, r.pp, r.budget) for r in rows.itertuples()])
    coef, *_ = np.linalg.lstsq(X, rows["overhead_gib"].to_numpy(), rcond=None)
    return coef


def file_based(df, invs):
    """Leave-one-model-out: overhead coefficients from the other models' engines."""
    ok = df[df["fits"] & df["overhead_gib"].notna()]
    pred = []
    for r in df.itertuples():
        inv = invs[r.model]
        coef = fit_overhead(ok[ok["model"] != r.model], invs)
        stages = inv.stage_gib(r.tp, r.pp, r.quantization)
        ov = float(overhead_features(inv, r.tp, r.pp, r.budget) @ coef)
        ov_last = float(overhead_features(inv, r.tp, 1, r.budget) @ coef) if r.pp > 1 else ov
        free = [BUDGET_GIB - w - (ov_last if s == len(stages) - 1 else ov) for s, w in enumerate(stages)]
        kv_free = min(free)
        tokens = kv_free * GIB / inv.kv_bytes_per_token(r.tp, r.pp) // BLOCK * BLOCK
        fits = kv_free > 0 and (inv.model_type in ATTN_ONLY or tokens >= r.max_seq_len)
        pred.append({"w0_pred": stages[0], "kv_tokens_pred": tokens if kv_free > 0 else 0.0,
                     "overhead_pred": ov, "fits_pred": fits})
    return pd.DataFrame(pred, index=df.index)


def maveriq_form(df):
    """M(t,p) = M11 + (t-1) A + t (p-1) A' on per-model totals (sum over GPUs, assuming
    balanced stages), fitted at each budget from its (1,1), (1,2), (2,1) engines."""
    out = pd.DataFrame(index=df.index, columns=["w0_mav", "mem_mav", "fits_mav"], dtype=object)
    for (m, q, b), g in df.groupby(["model", "quantization", "budget"]):
        ref = {(r.tp, r.pp): r for r in g.itertuples() if r.fits}
        if not all(k in ref for k in [(1, 1), (1, 2), (2, 1)]):
            continue
        for col, target in (("w0_mav", "weights_gib"), ("mem_mav", "nonkv")):
            val = {k: (ref[k].weights_gib if target == "weights_gib"
                       else ref[k].weights_gib + ref[k].overhead_gib) * k[0] * k[1] for k in ref}
            m11 = val[(1, 1)]
            A, Ap = max(val[(2, 1)] - m11, 0), max(val[(1, 2)] - m11, 0)
            for r in g.itertuples():
                out.at[r.Index, col] = (m11 + (r.tp - 1) * A + r.tp * (r.pp - 1) * Ap) / (r.tp * r.pp)
        for r in g.itertuples():
            out.at[r.Index, "fits_mav"] = out.at[r.Index, "mem_mav"] < BUDGET_GIB
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    df = engines()
    invs = {m: Inventory(m) for m in df["model"].unique()}
    for m, inv in invs.items():
        print(f"{m:45s} {inv.params / 1e9:6.2f} B params, layers {inv.n_layers}, tied {inv.tied}, "
              f"kv heads {inv.kv_heads}, type {inv.model_type}")
    df = pd.concat([df, file_based(df, invs), maveriq_form(df)], axis=1)
    df["w0_naive"] = [invs[r.model].params * (0.53 if r.quantization == "bitsandbytes" else 2)
                      / (r.tp * r.pp) / GIB for r in df.itertuples()]
    df["fits_naive"] = df["w0_naive"] < BUDGET_GIB
    df.to_csv(os.path.join(OUT, "memory_predictions.csv"), index=False)

    ok = df[df["fits"] & df["weights_gib"].notna()]
    ape = lambda p, y: 100 * np.abs(p.astype(float) - y) / y
    print("\nWeights per GPU (stage 0), MAPE over finished engines:")
    for col, lab in [("w0_pred", "file-based"), ("w0_mav", "MaverIQ-form"), ("w0_naive", "naive")]:
        s = ok[ok[col].notna()]
        print(f"  {lab:13s} {ape(s[col], s['weights_gib']).mean():6.2f}%  (n={len(s)}, "
              f"models={s['model'].nunique()})")
    s = ok[ok["w0_mav"].notna()]
    print(f"  same {len(s)} engines: file-based {ape(s['w0_pred'], s['weights_gib']).mean():.2f}%, "
          f"MaverIQ-form {ape(s['w0_mav'], s['weights_gib']).mean():.2f}%")
    ov = ok[ok["overhead_gib"].notna()]
    print(f"Activation/runtime overhead: mean abs error {np.abs(ov['overhead_pred'] - ov['overhead_gib']).mean():.2f} GiB "
          f"(measured {ov['overhead_gib'].min():.2f}-{ov['overhead_gib'].max():.2f} GiB)")
    kv = ok[ok["kv_tokens"].notna() & ~ok["model"].map(lambda m: invs[m].model_type in ATTN_ONLY)]
    print(f"KV cache tokens, file-based: MAPE {ape(kv['kv_tokens_pred'], kv['kv_tokens']).mean():.1f}% "
          f"(n={len(kv)})")
    print("\nFeasibility (engine starts or not):")
    for col, lab in [("fits_pred", "file-based"), ("fits_mav", "MaverIQ-form"), ("fits_naive", "naive")]:
        s = df[df[col].notna()]
        p, y = s[col].astype(bool), s["fits"].astype(bool)
        print(f"  {lab:13s} accuracy {100 * (p == y).mean():5.1f}%  n={len(s)}  "
              f"fits-but-fails {int((p & ~y).sum())}  fails-but-fits {int((~p & y).sum())}")
    kv_by = kv.assign(e=ape(kv["kv_tokens_pred"], kv["kv_tokens"])).groupby("model")["e"].mean()
    print(f"KV tokens per model: median {kv_by.median():.1f}%, range {kv_by.min():.1f}-{kv_by.max():.1f}%; "
          f"without InternVL2 {kv[~kv['model'].str.contains('InternVL')].pipe(lambda d: ape(d['kv_tokens_pred'], d['kv_tokens']).mean()):.1f}%")
    print(f"infeasible configs: {int((~df['fits']).sum())} of {len(df)}; with MaverIQ-form defined: "
          f"{int((~df.loc[df['fits_mav'].notna(), 'fits']).sum())}")
    by = ok.assign(err=ape(ok["w0_pred"], ok["weights_gib"]),
                   kv_err=ape(ok["kv_tokens_pred"], ok["kv_tokens"])).groupby(["model", "quantization"])
    summary = by[["err", "kv_err"]].mean().round(2)
    summary.to_csv(os.path.join(OUT, "memory_by_model.csv"))
    print(summary.to_string())


if __name__ == "__main__":
    main()
