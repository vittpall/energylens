"""Online replay of Azure traces: requests arrive over time into a running engine.

The trace mode (trace_workload.py) runs each batch of real requests offline, all
at once. Here the same requests (real input/output lengths, from the Azure LLM
inference traces) are sent at their trace arrival times, scaled by a rate factor
like MaverIQ's trace generator (rate 0.1 = a tenth of the trace's requests per
second), into one engine that keeps running: vLLM's continuous batching admits
each request when it arrives, so batches form, grow and shrink on their own, and
the GPUs idle between requests. Energy is measured with Zeus over the whole
window, from the first arrival until the last request finishes, so it includes
that idle time.

One window = `window_s` seconds of arrivals from a random offset of one trace at
one rate. Each (TP, PP) layout replays the same windows, so a window is a
scenario. Per window a CSV row (energy, duration, load), and per request its
send, first-token and finish times (requests.jsonl), from which the in-flight
count over time can be rebuilt. Each layout also gets an idle-power reading.
"""
import csv
import gc
import json
import os
import time

import numpy as np
import pandas as pd

from .trace_workload import token_pool, make_prompts

FIELDS = ["trace", "window_id", "rate", "hour", "n_requests", "arrival_span_s", "duration_s",
          "latency", "power", "energy", "idle_power", "n_gpus_measured", "power_source",
          "sum_input", "sum_output", "mean_input", "mean_output", "max_input", "max_output",
          "mean_in_flight", "max_in_flight", "mean_ttft_s", "p90_ttft_s", "mean_e2e_s", "p90_e2e_s",
          "tensor_parallel_size", "pipeline_parallel_size", "max_num_batched_tokens", "max_num_seqs"]


def build_windows(trace_paths, rates, windows_per, window_s, max_model_len, max_output, seed, hours=None):
    """Request lists with arrival offsets (s) in wall time: a slice of
    window_s * rate seconds of trace time, stretched by 1 / rate. With `hours`, one
    window per trace and hour of the day starting inside that hour (a daily load
    pattern at a fixed rate scale) instead of `windows_per` at random offsets."""
    rng = np.random.RandomState(seed)
    windows, wid = [], 0
    for path in trace_paths:
        t = pd.read_csv(path, usecols=["TIMESTAMP", "ContextTokens", "GeneratedTokens"])
        ts = pd.to_datetime(t["TIMESTAMP"], format="mixed", utc=True)
        t["ts"] = (ts - ts.min()).dt.total_seconds()
        t["GeneratedTokens"] = t["GeneratedTokens"].clip(1, max_output)
        t = t[t["ContextTokens"].between(1, max_model_len - t["GeneratedTokens"] - 8)].reset_index(drop=True)
        name = os.path.splitext(os.path.basename(path))[0].replace("AzureLLMInferenceTrace_", "").replace("_day", "")
        if hours is not None:
            for rate in rates:
                span = window_s * rate
                for h in hours:
                    start = rng.uniform(h * 3600, (h + 1) * 3600 - span)
                    r = t[(t["ts"] >= start) & (t["ts"] < start + span)]
                    if len(r) < 2:
                        continue
                    windows.append({"window_id": wid, "trace": name, "rate": rate, "hour": h,
                                    "arrivals": ((r["ts"] - start) / rate).round(4).tolist(),
                                    "inputs": r["ContextTokens"].astype(int).tolist(),
                                    "outputs": r["GeneratedTokens"].astype(int).tolist()})
                    wid += 1
            continue
        for rate in rates:
            span = window_s * rate                      # trace seconds per window
            found = 0
            for _ in range(50 * windows_per):            # the code trace has long gaps: resample
                if found == windows_per:
                    break
                start = rng.uniform(0, t["ts"].max() - span)
                r = t[(t["ts"] >= start) & (t["ts"] < start + span)]
                if len(r) < 2:
                    continue
                found += 1
                windows.append({"window_id": wid, "trace": name, "rate": rate,
                                "arrivals": ((r["ts"] - start) / rate).round(4).tolist(),
                                "inputs": r["ContextTokens"].astype(int).tolist(),
                                "outputs": r["GeneratedTokens"].astype(int).tolist()})
                wid += 1
    return windows


def in_flight(recs, duration):
    """Time-averaged and peak number of requests between send and finish."""
    ev = sorted([(r["sent"], 1) for r in recs] + [(r["done"], -1) for r in recs])
    n = area = peak = 0
    last = 0.0
    for t, d in ev:
        area += n * (t - last)
        n += d
        peak = max(peak, n)
        last = t
    return area / max(duration, 1e-9), peak


def replay(engine, window, prompts):
    """Send each request at its arrival time and step the engine until all finish.
    Returns per-request records (seconds from the window start)."""
    from vllm import SamplingParams
    n = len(window["arrivals"])
    params = [SamplingParams(max_tokens=o, min_tokens=o, ignore_eos=True, temperature=0.0)
              for o in window["outputs"]]
    recs = [{"input": window["inputs"][i], "output": window["outputs"][i],
             "arrival": window["arrivals"][i], "sent": None, "first": None, "done": None} for i in range(n)]
    ids = {}
    t0 = time.perf_counter()
    i = 0
    while i < n or engine.has_unfinished_requests():
        now = time.perf_counter() - t0
        while i < n and window["arrivals"][i] <= now:
            rid = f"w{window['window_id']}-{i}"
            engine.add_request(rid, prompts[i], params[i])
            recs[i]["sent"] = now
            ids[rid] = i
            i += 1
        if engine.has_unfinished_requests():
            outs = engine.step()
            t = time.perf_counter() - t0
            for o in outs:
                r = recs[ids[o.request_id]]
                if r["first"] is None and o.outputs and len(o.outputs[0].token_ids) > 0:
                    r["first"] = t
                if o.finished:
                    r["done"] = t
        elif i < n:
            time.sleep(min(max(window["arrivals"][i] - now, 0.0), 0.05))
    return recs


def profile_online(model_name, trace_paths, output_dir, tp_values, pp_values, rates=(0.1, 0.3, 1.0),
                   windows_per=2, window_s=60.0, max_output=1024, max_model_len=8192, budget=8192,
                   seed=0, trust_remote_code=True, extra_engine_args=None, idle_s=10.0, max_restarts=3,
                   seqs_values=None, budgets=None, hours=None):
    """One engine per configuration (layout, concurrency limit max_num_seqs, token budget)
    replays every window; rows are written per window, so a resubmit resumes where it
    stopped. With one concurrency limit and one budget the configuration key is the
    layout alone (tp<t>_pp<p>), as in earlier runs."""
    from transformers import AutoTokenizer
    from .model import Model, DEFAULT_MAX_NUM_SEQS
    from .profiler import Profiler
    from .gpu_guard import assert_gpus_exclusive
    from .profiling import _accel, load_model_meta

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "profiling_metrics.csv")
    win_path = os.path.join(output_dir, "windows.json")
    req_path = os.path.join(output_dir, "requests.jsonl")
    progress_path = os.path.join(output_dir, "progress.json")
    if os.path.exists(win_path):
        windows = json.load(open(win_path))
    else:
        windows = build_windows(trace_paths, rates, windows_per, window_s, max_model_len, max_output, seed, hours)
        json.dump(windows, open(win_path, "w"))
    progress = json.load(open(progress_path)) if os.path.exists(progress_path) else {}
    progress.setdefault("done", {})
    progress.setdefault("failed", [])
    progress.setdefault("crashes", {})
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(FIELDS)
    save = lambda: json.dump(progress, open(progress_path, "w"), indent=1)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    pool = token_pool(tokenizer)
    meta = load_model_meta(model_name, trust_remote_code)
    n_gpus = _accel().device_count()
    for w in windows:
        print(f"[online] window {w['window_id']}: {w['trace']} x{w['rate']}, {len(w['arrivals'])} requests "
              f"over {w['arrivals'][-1]:.0f} s", flush=True)

    seqs_values = list(seqs_values or [DEFAULT_MAX_NUM_SEQS])
    budgets = list(budgets or [budget])
    single = len(seqs_values) == 1 and len(budgets) == 1
    combos = [(tp, pp, sq, bg) for tp in tp_values for pp in pp_values for sq in seqs_values for bg in budgets]
    for tp, pp, seqs, budget in combos:
            cfg = f"tp{tp}_pp{pp}" if single else f"tp{tp}_pp{pp}_s{seqs}_b{budget}"
            if tp * pp > n_gpus or cfg in progress["failed"]:
                continue
            if meta["has_attention"] and meta["num_attention_heads"] % tp:
                continue
            done = progress["done"].setdefault(cfg, [])
            crashes = progress["crashes"].setdefault(cfg, {})
            for attempt in range(max_restarts + 1):
                # vLLM's TP=2 engine sometimes dies with an illegal memory access (as in
                # the offline trace runs): reload it and go on; a window that kills the
                # engine twice is skipped.
                todo = [w for w in windows if w["window_id"] not in done
                        and crashes.get(str(w["window_id"]), 0) < 2]
                if not todo:
                    break
                assert_gpus_exclusive(f"before loading {cfg}")
                model = engine = prof = current = None
                try:
                    # The watchdog bounds one window: arrivals + a drain of up to 10x the span.
                    model = Model(name=model_name, max_model_len=max_model_len,
                                  trust_remote_code=trust_remote_code, tensor_parallel_size=tp,
                                  pipeline_parallel_size=pp, max_num_batched_tokens=budget,
                                  enable_prefix_caching=False, dtype="bfloat16", max_num_seqs=seqs,
                                  generate_timeout_s=int(window_s * 10 + 600), **(extra_engine_args or {}))
                    engine = model.llm.llm_engine
                    prof = Profiler(handle=model.nvml_handle, gpu_indices=list(range(model.n_gpus_used)))
                    warm = {"window_id": -1, "arrivals": [0.0, 0.0], "inputs": [256, 256], "outputs": [16, 16]}
                    replay(engine, warm, make_prompts(warm, pool, seed=0))
                    _, idle = prof.profile(time.sleep, idle_s)
                    idle_power = idle["energy_joules"] / idle["total_latency"]
                    print(f"[online] {cfg}: idle {idle_power:.0f} W over {model.n_gpus_used} GPU(s)", flush=True)
                    for w in todo:
                        assert_gpus_exclusive(f"during {cfg}")
                        prompts = make_prompts(w, pool, seed=w["window_id"] + 1)
                        with model._generation_watchdog():
                            current = w["window_id"]
                        recs, m = prof.profile(replay, engine, w, prompts)
                        dur = m["total_latency"]
                        mean_if, max_if = in_flight(recs, dur)
                        ttft = np.array([r["first"] - r["sent"] for r in recs])
                        e2e = np.array([r["done"] - r["sent"] for r in recs])
                        row = {"trace": w["trace"], "window_id": w["window_id"], "rate": w["rate"],
                               "hour": w.get("hour", ""),
                               "n_requests": len(recs), "arrival_span_s": w["arrivals"][-1],
                               "duration_s": dur, "latency": dur, "power": m["avg_power_watts"],
                               "energy": m["energy_joules"], "idle_power": idle_power,
                               "n_gpus_measured": m["n_gpus_measured"], "power_source": m["power_source"],
                               "sum_input": sum(w["inputs"]), "sum_output": sum(w["outputs"]),
                               "mean_input": np.mean(w["inputs"]), "mean_output": np.mean(w["outputs"]),
                               "max_input": max(w["inputs"]), "max_output": max(w["outputs"]),
                               "mean_in_flight": mean_if, "max_in_flight": max_if,
                               "mean_ttft_s": ttft.mean(), "p90_ttft_s": np.percentile(ttft, 90),
                               "mean_e2e_s": e2e.mean(), "p90_e2e_s": np.percentile(e2e, 90),
                               "tensor_parallel_size": tp, "pipeline_parallel_size": pp,
                               "max_num_batched_tokens": budget, "max_num_seqs": seqs}
                        with open(csv_path, "a", newline="") as f:
                            csv.writer(f).writerow([row[k] for k in FIELDS])
                        with open(req_path, "a") as f:
                            f.write(json.dumps({"cfg": cfg, "window_id": w["window_id"], "requests": recs}) + "\n")
                        done.append(w["window_id"])
                        save()
                        print(f"[online] {cfg} window {w['window_id']}: {len(recs)} req, {dur:.0f} s, "
                              f"{m['energy_joules'] / 1e3:.1f} kJ, in flight {mean_if:.1f} (max {max_if})", flush=True)
                except Exception as e:
                    print(f"[online] {cfg} failed (attempt {attempt + 1}): {type(e).__name__}: {e}", flush=True)
                    if type(e).__name__ in ("GPUContentionError", "EngineHangError"):
                        save()
                        raise
                    if current is None:          # the engine never served a window
                        if not done:
                            progress["failed"].append(cfg)
                        save()
                        break
                    crashes[str(current)] = crashes.get(str(current), 0) + 1
                    save()
                finally:
                    # Drop our references to the engine first: while `engine` is alive its
                    # core process keeps the GPU memory, and the next layout cannot start.
                    engine = prof = None
                    if model is not None:
                        model.close()
                    model = None
                    gc.collect()
                    _accel().empty_cache()
                    save()
                    time.sleep(10)
