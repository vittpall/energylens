"""Pre-download a HuggingFace model (+ tokenizer/config) into the shared HF cache.

Run this on a node with internet access (e.g. an Aurora login node), NOT via
qsub — compute nodes are offline and model.pbs runs with HF_HUB_OFFLINE=1.

Usage:
    python download_model.py
    python download_model.py --model Qwen/Qwen2.5-14B-Instruct
"""

import argparse
import os

os.environ.setdefault("HF_HOME", "models/hf_cache")
os.environ.setdefault(
    "HF_HUB_CACHE", "models/hf_cache/hub"
)

from huggingface_hub import snapshot_download

DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"  # keep in sync with main_profiler.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    print(f"Downloading {args.model} into {os.environ['HF_HUB_CACHE']} ...")
    # the shared filesystem was nearly full: skip the duplicate weight formats repos ship
    # (original/ + *.pth, consolidated*.safetensors, and *.bin next to safetensors).
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(args.model)
    ignore = ["original/*", "*.pth", "*.gguf", "consolidated*.safetensors", "*.msgpack", "*.h5", "*.onnx"]
    if any(f.endswith(".safetensors") for f in files):
        ignore += ["*.bin", "*.pt"]
    path = snapshot_download(
        repo_id=args.model,
        cache_dir=os.environ["HF_HUB_CACHE"],
        ignore_patterns=ignore,
    )
    print(f"Done. Cached at: {path}")


if __name__ == "__main__":
    main()
