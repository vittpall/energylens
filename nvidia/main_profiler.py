"""Entry point for the profiling pipeline with multi-modal support.

The model, modality and sweep are chosen on the command line, so several jobs
can be queued at once without editing this file between submissions (a queued
PBS job reads this file only when it starts, so editing it would change every
job still waiting in the queue). model.pbs lists one line per model to profile.

    python main_profiler.py --model tiiuae/falcon-mamba-7b
    python main_profiler.py --model Qwen/Qwen2-VL-7B-Instruct --modality text_image
    python main_profiler.py --model mistralai/Mistral-7B-Instruct-v0.3 --quantization bitsandbytes
    python main_profiler.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --tp 1 2 4 8 --pp 1 2 4 8
    python main_profiler.py --model mistralai/Mistral-7B-Instruct-v0.3 --input-lengths 128 512 2048 4096
    python main_profiler.py --model mistralai/Mistral-7B-Instruct-v0.3 \
        --trace ../data/azure_traces/AzureLLMInferenceTrace_conv.csv ../data/azure_traces/AzureLLMInferenceTrace_code.csv

With no arguments it profiles Mistral-Small-24B text-only in fp16, the
configuration this file had hard-coded before the flags existed.

Output goes to ./results_zeus/run_<modality>_<model>_<fp16|bnb4bit>/<res_tag>/
(--trace: ./results_zeus_trace/run_trace_<model>_<fp16|bnb4bit>/, see
source/trace_workload.py; --input-lengths: ./results_zeus_inlen/.../in<L>/, merged
into .../in_all/profiling_metrics.csv).
Only one job may write a given output folder at a time: profiling.py appends
to the CSV and keeps a resume file there, so a second job on the same model
would interleave rows. A lock file enforces this (see _acquire_output_lock).
"""
import os

# Environment setup (must be before imports)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ['RAY_TMPDIR'] = '/tmp'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_TRUST_REMOTE_CODE'] = '1'
# expandable_segments lets the CUDA caching allocator grow segments instead of
# needing a single contiguous block. The InternVL-76B PP8 run OOM'd in the
# vision encoder with ~966 MiB reserved-but-unallocated (fragmentation) while
# trying to allocate 776 MiB contiguous — exactly the case PyTorch recommends
# this flag for. Set both the CUDA-specific and backend-agnostic env names.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
# Intel GPUs (Aurora): Level Zero Sysman must be enabled before the first Level
# Zero initialisation (torch.xpu / vLLM) for the energy counters in
# source/energy_intel.py. Ignored on NVIDIA.
os.environ.setdefault('ZES_ENABLE_SYSMAN', '1')

import argparse
import atexit
import random
import socket
import subprocess
import sys

from datasets import load_dataset
from source.dataloader import Dataloader
from source.model import EngineHangError
from source.gpu_guard import assert_gpus_exclusive, GPUContentionError
from source.profiling import profile_workloads, Modality
from source.trace_workload import profile_trace


# ════════════════════════════════════════════════════════════════════════
# PER-MODALITY DEFAULTS
# ════════════════════════════════════════════════════════════════════════
# Text keeps the exact grid of the existing results_zeus/ runs so new text runs
# stay comparable with them. Image/video get larger token budgets: each item is
# hundreds to thousands of tokens, so a 1k-16k budget packs only a handful of
# items per batch and the old runs ended up with 2-8 distinct batch sizes
# (8-32 scenarios, vs ~200 for text).
TEXT_BUDGETS = [1024, 2048, 4096, 8192, 16384]
MM_BUDGETS   = TEXT_BUDGETS + [32768, 65536]

MODALITY_DEFAULTS = {
    "text_only":  dict(budgets=TEXT_BUDGETS, n_items=1000, frames=0,  resolutions=[None]),
    "text_image": dict(budgets=MM_BUDGETS,   n_items=500,  frames=0,  resolutions=[None]),
    "text_video": dict(budgets=MM_BUDGETS,   n_items=200,  frames=16, resolutions=[240, 360, 480, 720]),
    "video_only": dict(budgets=MM_BUDGETS,   n_items=50,   frames=8,  resolutions=[240, 360, 480, 720]),
}


def fixed_length_dataset(model_name, length, n_items, seed=0):
    """n_items wikitext prompts of `length` tokens each (random windows of the
    tokenized train split), so input length becomes a grid dimension."""
    from datasets import Dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    text = "\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
    ids = tok(text, add_special_tokens=False).input_ids
    rng = random.Random(seed + length)
    starts = [rng.randrange(len(ids) - length) for _ in range(n_items)]
    return Dataset.from_list([{"text": tok.decode(ids[o:o + length])} for o in starts])


def checkpoint_precision(model):
    """Precision tag for the output folder: fp16, or the quantization a
    pre-quantized checkpoint declares (e.g. awq4bit, gptq8bit, w8a8)."""
    from transformers import AutoConfig
    q = getattr(AutoConfig.from_pretrained(model, trust_remote_code=True), "quantization_config", None)
    if not q:
        return "fp16"
    q = q if isinstance(q, dict) else q.to_dict()
    method = q.get("quant_method", "quant")
    if method == "compressed-tensors":
        g = next(iter(q.get("config_groups", {}).values()), {})
        w = (g.get("weights") or {}).get("num_bits")
        a = (g.get("input_activations") or {}).get("num_bits")
        return f"w{w}a{a}" if a else f"w{w}a16"
    return f"{method}{q.get('bits', '')}bit"


def merge_input_lengths(base_dir, lengths):
    import pandas as pd
    parts = [f"{base_dir}/in{L}/profiling_metrics.csv" for L in lengths]
    parts = [pd.read_csv(p) for p in parts if os.path.exists(p)]
    if parts:
        os.makedirs(f"{base_dir}/in_all", exist_ok=True)
        pd.concat(parts, ignore_index=True).to_csv(f"{base_dir}/in_all/profiling_metrics.csv", index=False)
        print(f"[input-lengths] merged {len(parts)} runs -> {base_dir}/in_all/profiling_metrics.csv")


def _video_mme(n_items):
    """Video-MME questions on short clips (< 2 min), one question per video.

    Video-MME stores YouTube IDs; the clips come from the dataset's own zip chunks,
    pre-extracted to source.profiling.VIDEO_CACHE_DIR as <videoID>.mp4 (YouTube
    downloads fail on the compute nodes). Only clips present there are used, so a
    missing clip cannot turn into an item without video. (sayakpaul/ucf101-subset,
    used before, has only 2 clips: every batch repeated them.)"""
    from source.profiling import VIDEO_CACHE_DIR
    ds = load_dataset("lmms-lab/Video-MME", split="test")
    seen, keep = set(), []
    for i, (vid, dur) in enumerate(zip(ds["videoID"], ds["duration"])):
        if dur == "short" and vid not in seen and os.path.exists(os.path.join(VIDEO_CACHE_DIR, f"{vid}.mp4")):
            seen.add(vid)
            keep.append(i)
    if len(keep) < n_items:
        print(f"[WARN] Video-MME: only {len(keep)} short clips cached in {VIDEO_CACHE_DIR} (wanted {n_items})")
    if not keep:
        sys.exit(f"[FATAL] no Video-MME clips in {VIDEO_CACHE_DIR}")
    return ds.select(keep[:n_items])


def load_modality_dataset(modality, n_items):
    """Dataset for a modality; n_items caps the split size."""
    if modality == "text_only":
        return load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[:{n_items}]")
    if modality == "text_image":
        return load_dataset("phiyodr/coco2017", split=f"validation[:{n_items}]",
                            trust_remote_code=True)
    if modality == "text_video":
        return _video_mme(n_items)
    else:  # video_only: the same Video-MME clips with the question removed, so a
        # request is the clip alone. (MSR-VTT, used before, only has YouTube links;
        # their downloads failed and ~90% of its rows ran without any video.)
        ds = _video_mme(n_items)
        return ds.remove_columns([c for c in ("question", "options", "answer") if c in ds.column_names])
    # Disable auto-decode of video column to avoid torchcodec/libnvrtc dependency.
    # _get_video_bytes() handles the raw {"bytes": ..., "path": ...} dict directly.
    if "video" in ds.column_names:
        from datasets import Video
        ds = ds.cast_column("video", Video(decode=False))
    return ds


# ════════════════════════════════════════════════════════════════════════
# OUTPUT-FOLDER LOCK
# ════════════════════════════════════════════════════════════════════════

def _job_is_running(owner):
    """True if the lock owner (a PBS job id, or pid@host) is still running."""
    if owner.startswith("pid") and "@" in owner:
        pid, host = owner[3:].split("@", 1)
        if host != socket.gethostname():
            return True          # can't check another host's pid — assume alive
        try:
            os.kill(int(pid), 0)
            return True
        except (ProcessLookupError, ValueError):
            return False
        except PermissionError:
            return True
    try:
        out = subprocess.run(["qstat", "-f", owner], capture_output=True,
                             text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True              # can't tell — refuse rather than risk mixing rows
    if out.returncode != 0:
        return False             # qstat no longer knows the job: it has finished
    state = [l.split("=", 1)[1].strip() for l in out.stdout.splitlines()
             if l.strip().startswith("job_state")]
    return bool(state) and state[0] in ("R", "E", "B")


def _acquire_output_lock(base_dir, force=False):
    """
    Refuse to start if another running job is writing `base_dir`.

    A stale lock (its job has finished, crashed, or hit walltime — os._exit
    skips the cleanup hook) is taken over, and profiling.py's resume file then
    continues where that job stopped.
    """
    os.makedirs(base_dir, exist_ok=True)
    lock = os.path.join(base_dir, ".profiling.lock")
    me = os.environ.get("PBS_JOBID") or f"pid{os.getpid()}@{socket.gethostname()}"
    if os.path.exists(lock):
        with open(lock) as f:
            owner = f.read().strip()
        if owner and owner != me:
            if _job_is_running(owner) and not force:
                sys.exit(f"[FATAL] {base_dir} is being written by running job "
                         f"{owner}. Two jobs on the same model would interleave "
                         f"rows in one CSV. Wait for it, or pass --force-lock if "
                         f"you are sure it is gone.")
            print(f"[info] taking over stale lock from {owner} — resuming")
    with open(lock, "w") as f:
        f.write(me)

    def _release():
        try:
            with open(lock) as f:
                if f.read().strip() == me:
                    os.remove(lock)
        except OSError:
            pass
    atexit.register(_release)


# ════════════════════════════════════════════════════════════════════════
# ARGUMENTS
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(
        description="Profile latency/energy of a model over a TP/PP x batch x "
                    "max_tokens grid. Unset options use the modality defaults.")
    ap.add_argument("--model", default="mistralai/Mistral-Small-24B-Instruct-2501",
                    help="Hugging Face model id")
    ap.add_argument("--modality", default="text_only", choices=list(MODALITY_DEFAULTS))
    ap.add_argument("--quantization", default="none", choices=["none", "bitsandbytes"],
                    help="none = full precision (fp16/bf16); bitsandbytes = 4-bit "
                         "nf4 in-flight quantization (attention models only — "
                         "Mamba checkpoints fail the bnb load)")
    ap.add_argument("--tp", type=int, nargs="+", default=[1, 2, 4],
                    help="tensor_parallel_size values (default 1 2 4). Configs "
                         "with TP*PP above the visible GPU count are skipped.")
    ap.add_argument("--pp", type=int, nargs="+", default=[1, 2, 4],
                    help="pipeline_parallel_size values (default 1 2 4)")
    ap.add_argument("--budgets", type=int, nargs="+", default=None,
                    help="max_num_batched_tokens values; they set the batch sizes "
                         "(default: text 1024..16384, image/video up to 65536)")
    ap.add_argument("--max-tokens", type=int, nargs="+", default=[64, 128, 256, 512],
                    help="output-length sweep (default 64 128 256 512)")
    ap.add_argument("--n-items", type=int, default=None,
                    help="dataset items to load (default: text 1000, image 500, "
                         "text+video 200, video-only 50)")
    ap.add_argument("--frames", type=int, default=None,
                    help="video frames per clip (default: text+video 16, video-only 8)")
    ap.add_argument("--resolutions", type=int, nargs="+", default=None,
                    help="video short-side resolutions to sweep "
                         "(default 240 360 480 720; ignored for text/image)")
    ap.add_argument("--max-num-seqs", type=int, nargs="+", default=None,
                    help="vLLM concurrency limits to sweep (heavy dimension). Default: not swept, "
                         "the engine uses source.model.DEFAULT_MAX_NUM_SEQS (64)")
    ap.add_argument("--samples-per-config", type=int, default=20)
    ap.add_argument("--max-model-len", type=int, default=None,
                    # InternVL tiles each image into up to ~13 patches (~256
                    # tokens each), so full-resolution image prompts reach ~3300
                    # tokens. 2048 was too small and caused ~688 generations to
                    # be skipped. 4096 fits them.
                    help="vLLM max_model_len (default 4096; 8192 with --trace)")
    ap.add_argument("--output-root", default=None,
                    help="default ./results_zeus (./results_zeus_trace with --trace)")
    ap.add_argument("--input-lengths", type=int, nargs="+", default=None,
                    help="text only: sweep prompt length (tokens per request) as a grid "
                         "dimension, e.g. 128 512 2048 4096. Prefix caching is turned off; "
                         "budgets default to 1024..65536 so long prompts still batch")
    ap.add_argument("--trace", nargs="+", default=None,
                    help="Azure trace CSVs: profile batches of real requests instead "
                         "of the dataset grid (text only; --budgets takes the first value)")
    ap.add_argument("--online", action="store_true",
                    help="with --trace: replay the requests at their arrival times into a "
                         "running engine (source/online_trace.py) instead of as offline batches")
    ap.add_argument("--online-rates", type=float, nargs="+", default=[0.1, 0.3, 1.0],
                    help="arrival-rate factors (1 = the trace's own rate)")
    ap.add_argument("--online-windows", type=int, default=2, help="windows per trace and rate")
    ap.add_argument("--online-window-s", type=float, default=60.0, help="seconds of arrivals per window")
    ap.add_argument("--online-seqs", type=int, nargs="+", default=None,
                    help="concurrency limits (max_num_seqs) to replay, each a separate configuration")
    ap.add_argument("--online-hours", type=int, nargs="+", default=None,
                    help="one window per trace starting inside each of these hours of the day "
                         "(use with a one-day trace, e.g. *_day.csv) instead of random offsets")
    ap.add_argument("--trace-batches", type=int, default=40)
    ap.add_argument("--trace-max-bs", type=int, default=64)
    ap.add_argument("--trace-max-output", type=int, default=1024,
                    help="cap on a request's output tokens")
    ap.add_argument("--trace-seed", type=int, default=0)
    ap.add_argument("--force-lock", action="store_true",
                    help="start even if the output folder's lock names a job "
                         "that still looks alive")
    return ap.parse_args()


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def run_trace(args, quant, quant_tag):
    if args.online:
        out = f"{args.output_root or './results_zeus_online'}/run_online_{args.model.split('/')[-1]}_{quant_tag}"
    else:
        out = f"{args.output_root or './results_zeus_trace'}/run_trace_{args.model.split('/')[-1]}_{quant_tag}"
    _acquire_output_lock(out, force=args.force_lock)
    try:
        assert_gpus_exclusive("at job start")
        if args.online:
            from source.online_trace import profile_online
            return profile_online(args.model, args.trace, out, args.tp, args.pp,
                                  rates=args.online_rates, windows_per=args.online_windows,
                                  window_s=args.online_window_s, max_output=args.trace_max_output,
                                  max_model_len=args.max_model_len or 8192,
                                  budget=(args.budgets or [8192])[0], seed=args.trace_seed,
                                  extra_engine_args={"quantization": quant} if quant else None,
                                  seqs_values=args.online_seqs, budgets=args.budgets, hours=args.online_hours)
        profile_trace(args.model, args.trace, out, args.tp, args.pp,
                      n_batches=args.trace_batches, max_bs=args.trace_max_bs,
                      max_output=args.trace_max_output,
                      max_model_len=args.max_model_len or 8192,
                      budget=(args.budgets or [8192])[0], seed=args.trace_seed,
                      extra_engine_args={"quantization": quant} if quant else None)
    except GPUContentionError as e:
        print(f"\n[FATAL] {e}", flush=True)
        sys.exit(3)
    except EngineHangError as e:
        print(f"\nFATAL: {e}", flush=True)
        os._exit(1)


def main():
    """Run the profiling pipeline."""
    args = parse_args()
    d = MODALITY_DEFAULTS[args.modality]
    modality = Modality[args.modality.upper()]
    budgets = args.budgets or d["budgets"]
    n_items = args.n_items or d["n_items"]
    num_video_frames = d["frames"] if args.frames is None else args.frames
    resolutions = (args.resolutions or d["resolutions"]) if d["resolutions"] != [None] else [None]
    quant = None if args.quantization == "none" else args.quantization
    quant_tag = {None: checkpoint_precision(args.model), "bitsandbytes": "bnb4bit"}[quant]
    if args.trace:
        return run_trace(args, quant, quant_tag)
    lengths = args.input_lengths
    if lengths and args.modality != "text_only":
        sys.exit("[FATAL] --input-lengths is for --modality text_only")
    if lengths:
        budgets = args.budgets or MM_BUDGETS
    max_model_len = args.max_model_len or (max(lengths) + max(args.max_tokens) + 64 if lengths else 4096)
    output_root = args.output_root or ("./results_zeus_inlen" if lengths else "./results_zeus")

    heavy_defs = [
        {
            "tensor_parallel_size": args.tp,
            "pipeline_parallel_size": args.pp,
            # Required, not optional: profiling.py only batches when this key
            # (or max_num_seqs) is present in heavy_cfg — otherwise it falls
            # through to one item per call and every row gets batch_size=1,
            # which makes the 1/(bs^beta + eps) terms unidentifiable.
            "max_num_batched_tokens": budgets,
        }
    ]
    if args.max_num_seqs:
        heavy_defs[0]["max_num_seqs"] = args.max_num_seqs
    if lengths:
        # The grid reuses the same prompts for every max_tokens value on one
        # engine; with long prompts, cached prefill would hide their energy.
        heavy_defs[0]["enable_prefix_caching"] = [False]
    if quant is not None:
        # "bitsandbytes" = 4-bit nf4 in-flight quantization of the unquantized
        # checkpoint (vLLM auto-sets load_format for it). 8-bit is omitted:
        # vLLM 0.11 has no int8 in-flight path.
        heavy_defs[0]["quantization"] = [quant]

    light_defs = [{"max_tokens": args.max_tokens}]

    USE_EXHAUSTIVE = True
    BASE_OUTPUT_DIR = (f"{output_root}/run_{args.modality}_"
                       f"{args.model.split('/')[-1]}_{quant_tag}")
    _acquire_output_lock(BASE_OUTPUT_DIR, force=args.force_lock)
    try:
        assert_gpus_exclusive("at job start")
    except GPUContentionError as e:
        sys.exit(f"[FATAL] {e}")

    ds = None if lengths else load_modality_dataset(args.modality, n_items)

    loader = Dataloader(
        heavy_defs=heavy_defs,
        light_defs=light_defs,
        n_heavy=10,
        n_light=10,
        method="random",
        exhaustive=USE_EXHAUSTIVE
    )

    print(f"Model: {args.model}")
    print(f"Mode: {'EXHAUSTIVE' if USE_EXHAUSTIVE else 'SAMPLED'}")
    print(f"Modality: {modality.name}  ({n_items} items, {num_video_frames} video frames)"
          + (f"  input lengths {lengths}, max_model_len {max_model_len}" if lengths else ""))
    print(f"TP={args.tp}  PP={args.pp}  budgets={budgets}  max_tokens={args.max_tokens}  "
          f"precision={quant_tag}")
    print(f"Generated {len(loader.heavy_configs)} heavy × {len(loader.light_configs)} light configs")
    print(f"Resolutions to sweep: {resolutions}" if not lengths else f"Input lengths: {lengths}")
    print(f"Output: {BASE_OUTPUT_DIR}")

    try:
        if lengths:
            runs = [(f"in{L}", lambda L=L: fixed_length_dataset(args.model, L, n_items), None)
                    for L in lengths]
        else:
            runs = [(f"res{r}p" if r is not None else "res_native", lambda: ds, r)
                    for r in resolutions]
        for tag, make_ds, res in runs:
            output_dir = f"{BASE_OUTPUT_DIR}/{tag}"
            print(f"\n{'='*60}\nRun: {tag}\n{'='*60}")
            profile_workloads(
                ds=make_ds(),
                heavy_configs=loader.heavy_configs,
                light_configs=loader.light_configs,
                model_name=args.model,
                modality=modality,
                num_samples_per_config=args.samples_per_config,
                trust_remote_code=True,
                max_model_len=max_model_len,
                num_video_frames=num_video_frames,
                output_dir=output_dir,
                target_video_resolution=res,
            )
            if lengths:
                merge_input_lengths(BASE_OUTPUT_DIR, lengths)
    except GPUContentionError as e:
        # Another job is on our GPUs (see source/gpu_guard.py). The model was
        # closed by profiling.py's finally block and the partial rows of the
        # interrupted config were removed, so a resubmit resumes cleanly.
        print(f"\n[FATAL] {e}", flush=True)
        sys.exit(3)
    except EngineHangError as e:
        # The vLLM engine deadlocked (a worker died, e.g. CUDA OOM) and the
        # generate() watchdog fired. Progress is checkpointed per-config, so a
        # resubmit resumes from here. Hard-exit to kill the wedged engine
        # subprocesses immediately instead of letting atexit hooks hang trying
        # to join them — which is what made the job idle until walltime before.
        # (That also leaves the output lock behind; the next job sees this one
        # is no longer running and takes it over.)
        print(f"\nFATAL: {e}", flush=True)
        os._exit(1)


if __name__ == "__main__":
    main()
