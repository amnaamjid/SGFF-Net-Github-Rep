#!/usr/bin/env python3
"""
inspect_duplicates.py
=======================
Run this BEFORE deleting anything based on duplicate_report.csv.

2,812 cross-pool matches between datasets built from totally different real
photo sources (IMDb-WIKI vs CelebA-HQ) is suspiciously high -- it's more
likely that face-crop/resize preprocessing made many DIFFERENT people's
photos look structurally similar to a loose perceptual hash, not that the
same photo genuinely appears twice.

WHAT THIS DOES
--------------
1. Reads duplicate_report.csv, filters to cross_split=True rows only.
2. Prints a breakdown of how many pairs came from each pool-pair combo, and
   the distribution of hamming_distance (real duplicates cluster near 0-2;
   if most of your 2,812 pairs are sitting at distance 4-5, that's a strong
   sign of false positives, not real duplicates).
3. Saves a handful of the CLOSEST matches (lowest distance = most likely to
   be genuinely the same photo) as side-by-side JPG contact sheets so you can
   eyeball them and confirm before deleting anything.

USAGE
-----
pip install pillow pandas --break-system-packages

python inspect_duplicates.py \
    --report duplicate_report.csv \
    --pool dff_c_train=/path/to/train/Real \
    --pool dff_c_val=/path/to/validation/Real \
    --pool dff_c_test=/path/to/test/Real \
    --pool dffd_a_train=/path/to/dffd_a/train/Real \
    --pool dffd_a_val=/path/to/dffd_a/val/Real \
    --pool dffd_a_test=/path/to/dffd_a/Test/Real \
    --pool dffd_b=/path/to/dffd_b/Real \
    --pool diffface=/path/to/diffface/Real \
    --samples 20 \
    --out inspect_samples

(use the SAME --pool name=path values you used for duplicate_detection.py)
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

try:
    import pandas as pd
    from PIL import Image
except ImportError:
    sys.exit("Missing dependency. Install with:\n  pip install pillow pandas --break-system-packages")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--pool", action="append", required=True, help="name=path, repeatable")
    ap.add_argument("--samples", type=int, default=20, help="How many closest pairs to export as images")
    ap.add_argument("--out", type=Path, default=Path("inspect_samples"))
    args = ap.parse_args()

    pools = {}
    for spec in args.pool:
        name, path = spec.split("=", 1)
        pools[name] = Path(path)

    df = pd.read_csv(args.report)
    cross = df[df["cross_split"] == True].copy()  # noqa: E712
    print(f"Total flagged pairs        : {len(df):,}")
    print(f"Cross-pool pairs           : {len(cross):,}")

    print("\nBreakdown by pool pair:")
    pair_counts = Counter(
        tuple(sorted([a, b])) for a, b in zip(cross["pool_a"], cross["pool_b"])
    )
    for (a, b), count in pair_counts.most_common():
        print(f"  {a:<14} <-> {b:<14}: {count:,}")

    print("\nHamming distance distribution (cross-pool pairs):")
    for d in sorted(cross["hamming_distance"].unique()):
        n = (cross["hamming_distance"] == d).sum()
        print(f"  distance {d}: {n:,} pairs")
    print(
        "\n  -> If most pairs sit at distance 4-5, treat with suspicion (likely false"
        "\n     positives from similar face-crop framing, not real duplicates)."
        "\n     If many sit at distance 0-2, those are very likely genuine duplicates."
    )

    # Export the closest (most likely genuine) pairs as side-by-side images
    args.out.mkdir(parents=True, exist_ok=True)
    closest = cross.sort_values("hamming_distance").head(args.samples)
    exported = 0
    for i, row in enumerate(closest.itertuples(), start=1):
        path_a = pools.get(row.pool_a, Path(".")) / row.file_a
        path_b = pools.get(row.pool_b, Path(".")) / row.file_b
        if not path_a.exists() or not path_b.exists():
            print(f"  WARNING: could not find {path_a} or {path_b}, skipping")
            continue
        try:
            img_a = Image.open(path_a).convert("RGB").resize((256, 256))
            img_b = Image.open(path_b).convert("RGB").resize((256, 256))
            combined = Image.new("RGB", (520, 256), "white")
            combined.paste(img_a, (0, 0))
            combined.paste(img_b, (264, 0))
            out_path = args.out / f"pair_{i:02d}_dist{row.hamming_distance}_{row.pool_a}_vs_{row.pool_b}.jpg"
            combined.save(out_path)
            exported += 1
        except Exception as e:
            print(f"  WARNING: could not process {path_a}/{path_b}: {e}")

    print(f"\nExported {exported} side-by-side sample images to: {args.out}/")
    print("Open a few of these and look with your own eyes: left image vs right image.")
    print("If they are clearly the SAME photo -> genuine duplicates, proceed to remove.")
    print("If they are clearly DIFFERENT people -> false positives, raise --threshold")
    print("stringency (e.g. rerun duplicate_detection.py with --threshold 2) instead of deleting.")


if __name__ == "__main__":
    main()
