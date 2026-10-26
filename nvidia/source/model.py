"""Model wrapper for vLLM with profiling capabilities."""
import base64
import io
import gc
import math
import os
import signal
try:
    import pynvml
except Exception:  # no NVIDIA bindings (Intel GPUs)
    pynvml = None
from contextlib import contextmanager
import numpy as np
from vllm import LLM, SamplingParams
from .profiler import Profiler


# Concurrency cap vLLM gets when the caller sets none. Batches larger than this
# run in waves; profiling.build_workload records it in the CSV's max_num_seqs.
DEFAULT_MAX_NUM_SEQS = 64

# Shortest measurement window (s). A batch that finishes faster is re-run inside
# one window of at least this length and its energy/latency divided by the
# repeat count (see generate). 0 disables. ENERGYLENS_MIN_WINDOW_S overrides.
MIN_WINDOW_S = float(os.environ.get("ENERGYLENS_MIN_WINDOW_S", "1.0"))
MAX_REPEATS = 50


class EngineHangError(RuntimeError):
    """Raised when a generate call exceeds its timeout.

    A vLLM v1 worker dying (e.g. CUDA OOM in the vision encoder) does not crash
    the parent EngineCore — it deadlocks waiting on the dead worker's shared
    memory broadcast, printing "No available shared memory broadcast block" once
    a minute forever. The driver's generate() call then blocks indefinitely
    instead of raising, so the PBS job idles until walltime. This exception turns
    that hang into a fatal, fast failure. It is unrecoverable in-process.
    """
    pass


class Model:
    """Wrapper around vLLM with resource management and profiling."""

    def __init__(self, name, metrics=[], **kwargs):
        self.name = name
        # Extract max_model_len from kwargs with default fallback
        max_model_len = kwargs.pop("max_model_len", 8192)
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        # Hard wall-clock limit (seconds) for a single generate() call. A healthy
        # batch finishes in seconds-to-minutes; anything longer means the engine
        # has deadlocked, so we raise EngineHangError instead of hanging forever.
        # 0/None disables the watchdog.
        self.generate_timeout_s = kwargs.pop("generate_timeout_s", 600)
        self.min_window_s = kwargs.pop("min_window_s", MIN_WINDOW_S)
        # Fraction of GPU memory vLLM may use for weights + KV cache.
        # Must leave room for the model weights themselves; a value too low
        # (e.g. 0.6 on a 40 GiB A100 with a 14B model) leaves negative KV
        # cache memory and the engine fails to start. Configurable per-run.
        gpu_memory_utilization = kwargs.pop("gpu_memory_utilization", 0.9)

        # Reduce max_num_seqs to decrease memory pressure during sampler warmup
        # Default 256 can cause OOM - reduce to 64-128 for safety
        if "max_num_seqs" not in kwargs:
            kwargs["max_num_seqs"] = DEFAULT_MAX_NUM_SEQS

        # On multi-GPU runs vLLM's custom all-reduce kernel fails on this hardware
        # (custom_all_reduce.cuh:455 'invalid argument' during warm-up, killing the
        # worker), because the GPUs lack a usable P2P/NVLink path. Fall back to NCCL
        # for tensor-parallel runs unless the caller explicitly overrides it.
        if kwargs.get("tensor_parallel_size", 1) > 1:
            kwargs.setdefault("disable_custom_all_reduce", True)

        # Before vLLM's V1 engine (0.8) chunked prefill is off by default, and the
        # budget sweep needs it (a budget below max_model_len is otherwise rejected).
        # V1 chunks prefill always, so this reproduces its behaviour on older builds.
        import vllm as _vllm
        _ver = tuple(int(x) for x in _vllm.__version__.split(".")[:2] if x.isdigit())
        if _ver < (0, 8):
            kwargs.setdefault("enable_chunked_prefill", True)

        # AMD MI100 (vLLM 0.6.3, RCCL over PCIe): with captured graphs a TP=2 engine
        # returns the right first token and garbage afterwards; in eager mode its output
        # matches TP=1 token for token (amd_setup/check_layout_output.py). Eager on every
        # layout so that layouts are compared under the same execution mode. RCCL P2P
        # across the two PCIe root complexes fails ("unhandled cuda error"), so it is off.
        from .energy_amd import amd_available
        if amd_available():
            kwargs.setdefault("enforce_eager", True)
            os.environ.setdefault("NCCL_P2P_DISABLE", "1")

        # ROCm on the MI100 node: the two GPUs sit on separate root complexes and
        # RCCL's P2P/IPC transport fails ('invalid device pointer'), so collectives
        # go through host shared memory. With HIP graphs those collectives return
        # garbage text at TP=2 (eager output matches TP=1 token for token), so graphs
        # are off on every layout to keep TP=1 and TP=2 on the same footing.
        import torch as _torch
        if getattr(_torch.version, "hip", None):
            os.environ.setdefault("NCCL_P2P_DISABLE", "1")
            kwargs.setdefault("enforce_eager", True)
            kwargs.setdefault("dtype", "float16")  # gfx908 has no bfloat16

        # GPUs this engine actually occupies. The job allocation is usually
        # larger, and vLLM ranks map onto the first tp*pp visible devices, so
        # energy must be measured over exactly these — measuring the whole
        # allocation would charge low-parallelism configs for idle GPUs.
        self.n_gpus_used = (kwargs.get("tensor_parallel_size", 1)
                            * kwargs.get("pipeline_parallel_size", 1))

        self.llm = LLM(
            model=name,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
            **kwargs
        )
        self.has_chat_template = False
        try:
            tokenizer = self.llm.get_tokenizer()
            self.has_chat_template = bool(getattr(tokenizer, "chat_template", None))
        except Exception:
            self.has_chat_template = False
        self.output_metrics = {}
        self.nvml_handle = self.init_nvml()

    def close(self):
        """Clean shutdown and free GPU resources."""
        # try LLM-level shutdowns
        for name in ("close", "shutdown", "stop", "terminate"):
            fn = getattr(self.llm, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

        # try engine-level shutdowns
        eng = getattr(self.llm, "llm_engine", None)
        if eng is not None:
            for name in ("shutdown", "close", "stop", "terminate"):
                fn = getattr(eng, name, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass

        # Clean up GPU memory
        try:
            del self.llm
        except Exception:
            pass

        # Older vLLM (0.6.x, the MI100 build) keeps its TP/PP groups after the
        # engine is gone, so the next config with another TP size fails with
        # "tensor parallel group already initialized". Tear them down explicitly.
        try:
            from vllm.distributed.parallel_state import (
                destroy_model_parallel, destroy_distributed_environment)
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass

        try:
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def generate(self, messages, sampling_params):
        """Generate output with profiling metrics.
        
        Args:
            messages: Either a single dict {"text": ..., "image": ...} or
                      a list of such dicts for batched inference.
            sampling_params: Dict of sampling parameters.
        """
        # Normalize input for shared routing logic.
        is_batch_input = isinstance(messages, list)
        message_batch = messages if is_batch_input else [messages]
        is_multimodal = any(
            m.get("image") is not None or m.get("video") is not None or m.get("video_frames") is not None
            for m in message_batch
        )

        # vLLM rejects empty decoder prompts, which can occur in text datasets
        # containing blank lines. Normalize to a minimal non-empty prompt.
        for m in message_batch:
            text = m.get("text", "")
            m["text"] = text if isinstance(text, str) and text.strip() else "."
        
        sampling_params = self._prepare_sampling_params(**sampling_params)

        profiler = Profiler(handle=self.nvml_handle,
                            gpu_indices=list(range(self.n_gpus_used)))
        if is_multimodal or self.has_chat_template:
            prepared_messages = [self._prepare_messages(**m) for m in message_batch]
            call = lambda: self.llm.chat(prepared_messages if is_batch_input else prepared_messages[0],
                                         sampling_params)
        else:
            prompts = [m.get("text", "") for m in message_batch]
            call = lambda: self.llm.generate(prompts if is_batch_input else prompts[0], sampling_params)

        with self._generation_watchdog():
            outputs, prof_metrics = profiler.profile(call)
            repeats = 1
            latency = prof_metrics.get("total_latency") or 0.0
            if self.min_window_s and latency < self.min_window_s:
                # Too short for the energy counter's update interval and the
                # 100 ms power sampling (validate_results.py: energy and power x
                # latency disagree almost only below ~0.5 s). Re-run the same
                # batch inside one longer window and divide by the repeat count;
                # the prefix cache is reset before each run so every repeat pays
                # its full prefill, like the single run it stands in for.
                repeats = min(MAX_REPEATS, math.ceil(self.min_window_s / max(latency, 1e-3)))

                def repeated():
                    runs = []
                    for _ in range(repeats):
                        self._reset_prefix_cache()
                        runs.append(call())
                    return runs

                runs, prof_metrics = profiler.profile(repeated)
                for k in ("total_latency", "energy_joules"):
                    if isinstance(prof_metrics.get(k), (int, float)):
                        prof_metrics[k] = prof_metrics[k] / repeats
                if isinstance(prof_metrics.get("energy_joules_per_gpu"), dict):
                    prof_metrics["energy_joules_per_gpu"] = {
                        g: e / repeats for g, e in prof_metrics["energy_joules_per_gpu"].items()}
                outputs = runs[-1]
        prof_metrics["measurement_repeats"] = repeats

        self.output_metrics.update(prof_metrics)

        # Handle batched outputs
        def longest(outs):
            outs = outs if isinstance(outs, list) else [outs]
            return [len(o.outputs[0].token_ids) for o in outs]
        lengths = longest(outputs)
        num_tokens = sum(lengths)
        generated_text = (outputs[0] if isinstance(outputs, list) else outputs).outputs[0].text
        # The batch decodes until its longest request stops, so its decode steps
        # (and energy) follow the longest output, not the sum or max_tokens.
        # With repeats (sampled outputs differ), the energy is the mean over the
        # runs, so the decode length is the mean of their longest outputs.
        if repeats > 1:
            self.output_metrics["max_generated_tokens"] = round(float(np.mean([max(longest(r)) for r in runs])))
        else:
            self.output_metrics["max_generated_tokens"] = max(lengths)

        return generated_text, num_tokens, self.output_metrics

    def _reset_prefix_cache(self):
        try:
            self.llm.reset_prefix_cache()
        except Exception:
            pass

    @contextmanager
    def _generation_watchdog(self):
        """Raise EngineHangError if the wrapped generate call runs too long.

        Uses SIGALRM (Unix, main thread only — which is where the profiling
        driver runs). When the engine deadlocks on a dead worker, the alarm
        interrupts the blocking IPC wait and surfaces a real exception instead
        of hanging until the PBS walltime expires.
        """
        timeout = self.generate_timeout_s
        if not timeout or timeout <= 0:
            yield
            return

        def _on_timeout(signum, frame):
            raise EngineHangError(
                f"generate() exceeded {timeout}s — vLLM engine likely deadlocked "
                "(a worker probably died, e.g. CUDA OOM). Aborting."
            )

        prev_handler = signal.signal(signal.SIGALRM, _on_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prev_handler)

    def _encode_image(self, path):
        """Encode local image file to base64."""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def _encode_pil_image(self, pil_img):
        """Encode PIL image to base64 JPEG."""
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()

    def _prepare_messages(self, text, image=None, video=None, video_frames=None):
        """Prepare multimodal messages for the model. Supports both text-only and multimodal."""
        # For text-only models or when no image is provided
        if image is None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ]
                }
            ]
        else:
            # Multimodal: text + image
            if isinstance(image, str):
                image_b64 = self._encode_image(image)
            else:  # PIL image
                image_b64 = self._encode_pil_image(image)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ]
        return messages

    def _prepare_sampling_params(self, **kwargs):
        """Convert config dict to vLLM SamplingParams."""
        # Remove keys that aren't valid SamplingParams
        kwargs.pop("batch_size", None)
        return SamplingParams(**kwargs)

    def init_nvml(self):
        """Initialize NVIDIA GPU monitoring (None on non-NVIDIA GPUs)."""
        try:
            pynvml.nvmlInit()
            return pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            return None

    def get_power_watts(self, handle):
        """Get GPU power draw in watts."""
        return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
