"""Build MaverIQ-style fingerprints: the model truncated to its first n decoder layers.

Same idea as MaverIQ's tools/profiling/fingerprint_generator.py (keep layers
0..n-1, set the layer count in the config), but done tensor by tensor from the
safetensors shards so no full model is loaded and any architecture whose layers
are named "...layers.<i>." works. Config entries with one value per layer (lists
or pattern strings such as layer_types, hybrid_override_pattern) are cut to n.
Writes <out>/<model>-<n>layer/ with a single model.safetensors, the rewritten
config and every other file of the snapshot (tokenizer, chat template, code),
except params.json (Mistral's native config, which keeps
the full layer count), plus fingerprint.json with the original layer count and the build time.

    python make_fingerprint.py --model Qwen/Qwen2.5-14B-Instruct --layers 1 2 3
"""
import argparse
import json
import os
import re
import shutil
import time

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

OUT = os.environ.get("FINGERPRINT_DIR", "models/fingerprints")
LAYER = re.compile(r"\blayers\.(\d+)\.")
COUNT_KEYS = ("num_hidden_layers", "n_layer", "num_layers")


def layer_count(cfg):
    for k in COUNT_KEYS:
        if isinstance(cfg.get(k), int):
            return k, cfg[k]
    raise ValueError(f"no layer count in config (looked for {COUNT_KEYS})")


def truncate_config(cfg, n):
    key, full = layer_count(cfg)
    out = dict(cfg)
    for k, v in cfg.items():
        if k in COUNT_KEYS and v == full:
            out[k] = n
        elif isinstance(v, (list, str)) and len(v) == full and k not in ("architectures",):
            out[k] = v[:n]
        elif isinstance(v, dict) and k == "text_config":
            out[k] = truncate_config(v, n)[0]
    return out, full


def build(model, n, out_root):
    src = snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt",
                                                   "*.py", "*.jinja", "*.tiktoken"])
    dst = os.path.join(out_root, f"{model.split('/')[-1]}-{n}layer")
    if os.path.exists(os.path.join(dst, "fingerprint.json")):
        print(f"[skip] {dst} exists")
        return dst
    start = time.time()
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "config.json")) as f:
        cfg, full = truncate_config(json.load(f), n)
    tensors = {}
    for shard in sorted(p for p in os.listdir(src)
                        if p.endswith(".safetensors") and not p.startswith("consolidated")):
        with safe_open(os.path.join(src, shard), framework="pt") as f:
            for name in f.keys():
                m = LAYER.search(name)
                if m is None or int(m.group(1)) < n:
                    tensors[name] = f.get_tensor(name)
    kept = {int(m.group(1)) for m in map(LAYER.search, tensors) if m}
    assert kept == set(range(n)), f"expected layers 0..{n - 1}, kept {sorted(kept)}"
    save_file(tensors, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
    for p in os.listdir(src):
        if p not in ("config.json", "params.json") and not p.endswith(".safetensors") and not p.endswith(".safetensors.index.json"):
            shutil.copy2(os.path.join(src, p), dst)   # resolves the hub's symlinks
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    info = {"model": model, "layers": n, "full_layers": full, "build_s": round(time.time() - start, 1)}
    with open(os.path.join(dst, "fingerprint.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[ok] {dst}  ({info['build_s']} s, {len(tensors)} tensors)")
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    for n in args.layers:
        build(args.model, n, args.out)


if __name__ == "__main__":
    main()
