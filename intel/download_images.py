"""Pre-download the COCO images the text_image profiling run uses.

The dataset (phiyodr/coco2017) holds only URLs and profiling.py fetched them at run
time, which fails on Aurora's offline compute nodes. Run this on a login node; the
images land in IMAGE_CACHE_DIR as <file name> and profiling._get_image reads them first.

    python download_images.py            # validation[:500], the text_image default
    python download_images.py --n 1000
"""
import argparse
import os
import sys
import urllib.request

os.environ.setdefault("HF_HOME", "models/hf_cache")
from datasets import load_dataset

CACHE = os.environ.get("IMAGE_CACHE_DIR", "models/image_cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    ds = load_dataset("phiyodr/coco2017", split=f"validation[:{args.n}]")
    done = failed = 0
    for item in ds:
        url = item["coco_url"]
        dst = os.path.join(CACHE, os.path.basename(url))
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            done += 1
            continue
        try:
            urllib.request.urlretrieve(url, dst + ".part")
            os.replace(dst + ".part", dst)
            done += 1
        except Exception as e:
            failed += 1
            print(f"[WARN] {url}: {e}", file=sys.stderr)
    print(f"{done}/{args.n} images in {CACHE} ({failed} failed)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
