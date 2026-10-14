"""Concatenate profiling_metrics.csv files found under a results folder.

    python concat_metrics.py [results_dir]
"""

import sys
import os
from pathlib import Path
import pandas as pd


CSV_NAME = "profiling_metrics.csv"
OUT_NAME = "combined_metrics.csv"


def collect_csvs(folder: Path, root: Path) -> pd.DataFrame:
    """Recursively find all CSVs under folder, tag with relative run path."""
    frames = []
    for csv_path in sorted(folder.rglob(CSV_NAME)):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  WARNING: could not read {csv_path}: {e}")
            continue
        run_tag = str(csv_path.parent.relative_to(root))
        df.insert(0, "run", run_tag)
        frames.append(df)
        print(f"  {csv_path.relative_to(root)}  ({len(df)} rows)")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./results")
    root = root.resolve()

    if not root.is_dir():
        print(f"ERROR: {root} is not a directory")
        sys.exit(1)

    print(f"Root: {root}\n")

    all_frames = []

    for subfolder in sorted(root.iterdir()):
        if not subfolder.is_dir():
            continue
        # Skip if this subfolder itself is just a single CSV (no nesting)
        csvs = list(subfolder.rglob(CSV_NAME))
        if not csvs:
            continue

        print(f"[{subfolder.name}]")
        df = collect_csvs(subfolder, root)
        if df.empty:
            print("  (no data)\n")
            continue

        out = subfolder / OUT_NAME
        df.to_csv(out, index=False)
        print(f"  -> {out.relative_to(root)}  ({len(df)} total rows)\n")
        all_frames.append(df)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True, sort=False)
        out = root / OUT_NAME
        combined.to_csv(out, index=False)
        print(f"Global -> {out.relative_to(root.parent)}  ({len(combined)} total rows)")
    else:
        print("No CSV files found.")


if __name__ == "__main__":
    main()
