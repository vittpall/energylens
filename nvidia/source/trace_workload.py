"""Profile batches of real Azure requests (input/output lengths from the trace)."""
import csv
import gc
import json
import os
import time

import numpy as np
import pandas as pd

FIELDS = ["trace", "batch_id", "batch_size", "input_length", "mean_input", "max_input",
          "max_tokens", "sum_output_tokens", "generated_tokens", "max_generated_tokens",
          "latency", "power", "energy", "n_gpus_measured", "power_source",
          "tensor_parallel_size", "pipeline_parallel_size", "max_num_batched_tokens",
          "max_num_seqs", "arrival_span_s"]


def build_batches(trace_paths, n_batches, max_bs, max_model_len, max_output, seed):
    """Batch specs split evenly over the traces; requests that do not fit
    max_model_len are dropped, outputs are capped at max_output."""
    rng = np.random.RandomState(seed)
    batches = []
    per_trace = np.array_split(np.arange(n_batches), len(trace_paths))
    for path, ids in zip(trace_paths, per_trace):
        t = pd.read_csv(path, usecols=["TIMESTAMP", "ContextTokens", "GeneratedTokens"])
        t["GeneratedTokens"] = t["GeneratedTokens"].clip(1, max_output)
        fits = t["ContextTokens"].between(1, max_model_len - t["GeneratedTokens"] - 8)
        print(f"[trace] {os.path.basename(path)}: {len(t)} requests, "
              f"{(~fits).mean():.1%} dropped (longer than max_model_len={max_model_len})")
        t = t[fits].reset_index(drop=True)
        name = os.path.splitext(os.path.basename(path))[0].replace("AzureLLMInferenceTrace_", "")
        for i in ids:
            bs = int(np.clip(np.round(np.exp(rng.uniform(0, np.log(max_bs)))), 1, max_bs))
            start = rng.randint(0, len(t) - bs)
            r = t.iloc[start:start + bs]
            ts = pd.to_datetime(r["TIMESTAMP"], format="mixed")
            batches.append({"batch_id": int(i), "trace": name,
                            "inputs": r["ContextTokens"].astype(int).tolist(),
                            "outputs": r["GeneratedTokens"].astype(int).tolist(),
                            "arrival_span_s": (ts.max() - ts.min()).total_seconds()})
    return batches


def token_pool(tokenizer, n_tokens=400_000):
    from datasets import load_dataset
    text = "\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return np.asarray(ids[:n_tokens])


def make_prompts(batch, pool, seed):
    rng = np.random.RandomState(seed)
    prompts = []
    for n in batch["inputs"]:
        start = rng.randint(0, len(pool) - n)
        prompts.append({"prompt_token_ids": pool[start:start + n].tolist()})
    return prompts


class EngineDead(RuntimeError):
    pass


def _run_batch(model, batch, pool):
    from vllm import SamplingParams
    from .profiler import Profiler
    from .model import EngineHangError
    prompts = make_prompts(batch, pool, seed=batch["batch_id"] + 1)
    params = [SamplingParams(max_tokens=o, min_tokens=o, ignore_eos=True, temperature=0.0)
              for o in batch["outputs"]]
    profiler = Profiler(handle=model.nvml_handle, gpu_indices=list(range(model.n_gpus_used)))
    try:
        with model._generation_watchdog():
            outputs, m = profiler.profile(model.llm.generate, prompts, params, use_tqdm=False)
    except EngineHangError:
        raise
    except Exception as e:
        raise EngineDead(f"batch {batch['batch_id']}: {type(e).__name__}: {e}") from e
    lengths = [len(o.outputs[0].token_ids) for o in outputs]
    bs = len(batch["inputs"])
    return {"generated_tokens": sum(lengths), "max_generated_tokens": max(lengths),
            "latency": m["total_latency"], "power": m["avg_power_watts"],
            "energy": m["energy_joules"] / bs, "n_gpus_measured": m["n_gpus_measured"],
            "power_source": m["power_source"]}


def profile_trace(model_name, trace_paths, output_dir, tp_values, pp_values,
                  n_batches=40, max_bs=64, max_output=1024, max_model_len=8192,
                  budget=8192, seed=0, trust_remote_code=True, extra_engine_args=None,
                  max_restarts=3):
    """Rows are written per finished batch, so a resubmit resumes mid-layout. If
    the engine dies on a batch (e.g. vLLM's TP=2 illegal memory access), that
    batch is skipped and the engine reloaded, up to max_restarts per layout."""
    import torch
    from transformers import AutoTokenizer
    from .model import Model, EngineHangError, DEFAULT_MAX_NUM_SEQS
    from .gpu_guard import assert_gpus_exclusive, GPUContentionError
    from .profiling import _accel, load_model_meta

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "profiling_metrics.csv")
    batches_path = os.path.join(output_dir, "batches.json")
    progress_path = os.path.join(output_dir, "progress.json")

    if os.path.exists(batches_path):
        batches = json.load(open(batches_path))
        print(f"[trace] reusing {len(batches)} batches from {batches_path}")
    else:
        batches = build_batches(trace_paths, n_batches, max_bs, max_model_len, max_output, seed)
        json.dump(batches, open(batches_path, "w"))
    progress = (json.load(open(progress_path)) if os.path.exists(progress_path) else {})
    for k, v in (("done", []), ("failed", []), ("batches", {}), ("skipped", {})):
        progress.setdefault(k, v)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(FIELDS)

    def save():
        json.dump(progress, open(progress_path, "w"), indent=1)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    pool = token_pool(tokenizer)
    meta = load_model_meta(model_name, trust_remote_code)
    n_gpus = _accel().device_count()
    sizes = [len(b["inputs"]) for b in batches]
    print(f"[trace] {len(batches)} batches, batch size {min(sizes)}-{max(sizes)} "
          f"(median {int(np.median(sizes))}), {sum(sizes)} requests per layout")

    for tp in tp_values:
        for pp in pp_values:
            cfg = f"tp{tp}_pp{pp}"
            if cfg in progress["done"] or tp * pp > n_gpus:
                continue
            if meta["has_attention"] and meta["num_attention_heads"] % tp:
                continue
            finished = progress["batches"].setdefault(cfg, [])
            skipped = progress["skipped"].setdefault(cfg, [])
            restarts = 0
            while True:
                todo = [b for b in batches if b["batch_id"] not in finished + skipped]
                if not todo:
                    progress["done"].append(cfg)
                    break
                assert_gpus_exclusive(f"before loading {cfg}")
                model = None
                try:
                    model = Model(name=model_name, max_model_len=max_model_len,
                                  trust_remote_code=trust_remote_code, tensor_parallel_size=tp,
                                  pipeline_parallel_size=pp, max_num_batched_tokens=budget,
                                  enable_prefix_caching=False, dtype="bfloat16",
                                  **(extra_engine_args or {}))
                    _run_batch(model, {"batch_id": -1, "inputs": [256, 256], "outputs": [16, 16]}, pool)
                    for b in todo:
                        assert_gpus_exclusive(f"during {cfg}")
                        try:
                            m = _run_batch(model, b, pool)
                        except EngineDead:
                            skipped.append(b["batch_id"])
                            raise
                        assert_gpus_exclusive(f"after batch {b['batch_id']} of {cfg}")
                        row = {"trace": b["trace"], "batch_id": b["batch_id"],
                               "batch_size": len(b["inputs"]), "input_length": sum(b["inputs"]),
                               "mean_input": np.mean(b["inputs"]), "max_input": max(b["inputs"]),
                               "max_tokens": max(b["outputs"]), "sum_output_tokens": sum(b["outputs"]),
                               "tensor_parallel_size": tp, "pipeline_parallel_size": pp,
                               "max_num_batched_tokens": budget, "max_num_seqs": DEFAULT_MAX_NUM_SEQS,
                               "arrival_span_s": b["arrival_span_s"], **m}
                        with open(csv_path, "a", newline="") as f:
                            csv.writer(f).writerow([row.get(k, "") for k in FIELDS])
                            f.flush()
                            os.fsync(f.fileno())
                        finished.append(b["batch_id"])
                        save()
                except (GPUContentionError, EngineHangError):
                    save()
                    raise
                except Exception as e:
                    print(f"[trace] {cfg} engine died: {type(e).__name__}: {e}", flush=True)
                    restarts += 1
                    if restarts > max_restarts or not finished and not isinstance(e, EngineDead):
                        progress["done"].append(cfg)
                        progress["failed"].append(cfg)
                        break
                finally:
                    if model is not None:
                        model.close()
                    gc.collect()
                    _accel().empty_cache()
                    save()
                    time.sleep(10)
            save()
