"""Profile MaverIQ fingerprints (the model truncated to a few layers) on real GPUs.

MaverIQ never runs the full model while profiling: for a reference layout with
PP stages it measures fingerprints of PP and PP+1 layers at two output lengths
and two batch sizes, and extrapolates per-layer TTFT/TPOT to the full layer
count (their tools/estimator/latency_estimator.py). This runs that protocol in
our setup: the same 3 reference layouts, middle token budget, shortest/longest
max_tokens and smallest/largest batch as profiling_cost.reference_rows, but on
fingerprints built by make_fingerprint.py. Requests ignore EOS so every one
decodes exactly max_tokens (a truncated model's EOS is meaningless; MaverIQ fixes
the output length the same way). Each row is measured --repeats times (MaverIQ's
run.py averages 10 iterations after 2 warm-ups): the per-layer extrapolation
multiplies the 1-vs-2-layer difference by ~the layer count, so single noisy
sub-second measurements extrapolate to nonsense, even negative latencies.

Output: fingerprint_results/<model>/profiling_metrics.csv (profiler format; its
num_layers column is the fingerprint's layer count) and engines.json (engine
start-up seconds per (TP, PP, layers), charged as profiling cost).

    python fingerprint_profiler.py --model Qwen/Qwen2.5-14B-Instruct \\
        --reference ../nvidia/results_zeus/run_text_only_Qwen2.5-14B-Instruct_fp16/res_native/profiling_metrics.csv
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NVIDIA = os.path.join(HERE, "..", "nvidia")
sys.path.insert(0, NVIDIA)
import main_profiler  # noqa: E402  (sets the vLLM/torch environment first)

import argparse  # noqa: E402
import csv  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from source.model import Model  # noqa: E402
from source.gpu_guard import assert_gpus_exclusive  # noqa: E402
from source.profiling import (  # noqa: E402
    Modality, load_model_meta, build_token_batches, build_workload, extract_item_data,
    init_csv_fields, init_csv_file, _build_model_message, _process_batch_and_write,
)
from make_fingerprint import OUT as FINGERPRINTS  # noqa: E402

MODALITY = Modality.TEXT_ONLY


def reference_layouts(reference):
    """The 3 cheapest layouts of the full grid (fewest GPUs, then PP), as profiling_cost.reference_rows."""
    d = pd.read_csv(reference)
    lay = d[["tensor_parallel_size", "pipeline_parallel_size"]].drop_duplicates().astype(int)
    lay = lay.assign(g=lay.tensor_parallel_size * lay.pipeline_parallel_size)
    lay = lay.sort_values(["g", "pipeline_parallel_size"]).head(3)
    budgets = sorted(d.max_num_batched_tokens.astype(int).unique())
    mts = sorted(d.max_tokens.astype(int).unique())
    # plain ints: vLLM's engine messages cannot serialise numpy integers
    return ([(int(t), int(p)) for t, p in zip(lay.tensor_parallel_size, lay.pipeline_parallel_size)],
            int(budgets[len(budgets) // 2]), (int(mts[0]), int(mts[-1])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--reference", required=True, help="the model's full Zeus grid CSV")
    ap.add_argument("--batches", type=int, default=20, help="candidate batches (smallest and largest are used)")
    ap.add_argument("--n-items", type=int, default=1000)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--repeats", type=int, default=10, help="measurements per row (averaged in the analysis)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    short = args.model.split("/")[-1]
    args.out = args.out or os.path.join(HERE, "fingerprint_results", short)
    main_profiler._acquire_output_lock(args.out)
    os.makedirs(args.out, exist_ok=True)

    layouts, budget, (mt_lo, mt_hi) = reference_layouts(args.reference)
    ds = main_profiler.load_modality_dataset("text_only", args.n_items)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    meta = load_model_meta(args.model, True, 0)
    batches = []
    for items, tokens in build_token_batches(ds, tokenizer, budget, MODALITY, model_config=meta["model_config"]):
        batches.append((items, tokens))
        if len(batches) == args.batches:
            break
    sizes = [len(b[0]) for b in batches]
    picked = [batches[sizes.index(min(sizes))], batches[sizes.index(max(sizes))]]
    print(f"[fp] layouts {layouts}  budget {budget}  max_tokens {mt_lo}/{mt_hi}  batch sizes {min(sizes)}/{max(sizes)}")

    csv_path = os.path.join(args.out, "profiling_metrics.csv")
    heavy_keys = [{"tensor_parallel_size": 0, "pipeline_parallel_size": 0, "max_num_batched_tokens": 0}]
    fields = init_csv_fields(heavy_keys, [{"max_tokens": 0}], {})
    init_csv_file(csv_path, fields)
    engines_path = os.path.join(args.out, "engines.json")
    engines = json.load(open(engines_path)) if os.path.exists(engines_path) else []
    done = {(e["tp"], e["pp"], e["layers"]) for e in engines}

    with open(csv_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        for tp, pp in layouts:
            for n in (pp, pp + 1):          # MaverIQ: PP and PP+1 layers, i.e. one stage gets one more layer
                if (tp, pp, n) in done:
                    print(f"[fp] skip ({tp},{pp}) {n} layers: done")
                    continue
                path = os.path.join(FINGERPRINTS, f"{short}-{n}layer")
                fp_meta = load_model_meta(path, True, 0)
                heavy = {"tensor_parallel_size": tp, "pipeline_parallel_size": pp,
                         "max_num_batched_tokens": budget, "dtype": "bfloat16"}
                assert_gpus_exclusive(f"before loading ({tp},{pp}) {n} layers")
                start = time.time()
                model = Model(name=path, max_model_len=args.max_model_len, trust_remote_code=True, **heavy)
                item = extract_item_data(ds[0], MODALITY, 0, fp_meta["model_config"])
                for _ in range(2):   # warm-up, as the grid and live profilers do
                    model.generate(messages=_build_model_message(item, MODALITY), sampling_params={"max_tokens": mt_lo})
                load_s = time.time() - start
                prof_s = 0.0
                for mt in (mt_lo, mt_hi):
                    for items, tokens in picked:
                        light = {"max_tokens": mt, "ignore_eos": True}
                        workload = build_workload(prompt=items[0]["text"], heavy_cfg=heavy, light_cfg=light,
                                                  modality=MODALITY, batch_size=len(items),
                                                  total_input_tokens=tokens, tokenizer=tokenizer)
                        assert_gpus_exclusive(f"({tp},{pp}) {n} layers")
                        t = time.time()
                        for _ in range(args.repeats):
                            _process_batch_and_write(model, items, MODALITY, light, heavy, workload,
                                                     fp_meta["model_info"], fields, writer, fh)
                        prof_s += time.time() - t
                model.close()
                del model
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(10)
                engines.append({"tp": tp, "pp": pp, "layers": n, "load_s": round(load_s, 1),
                                "profile_s": round(prof_s, 1)})
                json.dump(engines, open(engines_path, "w"), indent=2)
                print(f"[fp] ({tp},{pp}) {n} layers: load {load_s:.0f} s, 4 rows x {args.repeats} {prof_s:.0f} s")


if __name__ == "__main__":
    main()
