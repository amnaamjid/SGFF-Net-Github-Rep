#!/usr/bin/env python3
"""
duplicate_detection.py
=======================
Perceptual-hash (pHash) near-duplicate detector for Real image pools across
DFF (train/validation/test) and the cross-evaluation datasets (DiffFace, DFFD).

Identity-level splitting (see identity_split_pipeline.py) prevents leakage when
two images share the same *labeled* identity. It does NOT catch near-duplicate
images that exist under different filenames, possibly with missing/incorrect
identity metadata, or the same physical photo cropped/resized differently.
This script closes that gap.

WHAT IT DOES
------------
1. Walks one or more image pools you specify (each pool = a split, e.g.
   dff_train, dff_val, dff_test, diffface_real, dffd_real).
2. Computes a perceptual hash (pHash, 64-bit) for every image.
3. Finds all pairs of images (within the same pool AND across different pools)
   whose Hamming distance is below a threshold (default 5), which indicates
   they are near-duplicates or the same photo.
4. Flags any near-duplicate pair that crosses a split boundary (e.g. one image
   in dff_train and its near-duplicate in dff_test) as a LEAKAGE candidate.
5. Writes a CSV report of all flagged pairs for manual review, plus a summary.

USAGE
-----
pip install pillow imagehash --break-system-packages

python duplicate_detection.py \
    --pool dff_train=/path/to/DFF/Split/train/Real \
    --pool dff_val=/path/to/DFF/Split/validation/Real \
    --pool dff_test=/path/to/DFF/Split/test/Real \
    --pool diffface_real=/path/to/DiffFace/Real \
    --pool dffd_real=/path/to/DFFD/Real \
    --threshold 5 \
    --output duplicate_report.csv

Increase --threshold for a looser (more permissive) match, decrease for a
stricter one. 0 = exact hash match only. 5 is a reasonable default for
catching resized/recompressed copies of the same photo.
"""

import argparse
import csv
import itertools
import sys
from pathlib import Path

try:
    from PIL import Image
    import imagehash
except ImportError:
    sys.exit(
        "Missing dependencies. Install with:\n"
        "  pip install pillow imagehash --break-system-packages"
    )

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_pool_arg(value):
    """Parse --pool name=/path/to/dir into (name, Path)."""
    if "=" not in value:
        sys.exit(f"ERROR: --pool must be in the form name=/path, got: {value}")
    name, path_str = value.split("=", 1)
    path = Path(path_str)
    if not path.is_dir():
        sys.exit(f"ERROR: pool '{name}' path does not exist or is not a directory: {path}")
    return name, path


def compute_hashes(pools):
    """Return list of (pool_name, filename, phash) for every image in every pool."""
    records = []
    for pool_name, pool_path in pools.items():
        files = sorted(
            f for f in pool_path.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        print(f"[{pool_name}] hashing {len(files):,} images from {pool_path} ...")
        for i, f in enumerate(files, start=1):
            try:
                with Image.open(f) as img:
                    h = imagehash.phash(img)
                records.append((pool_name, f.name, h))
            except Exception as e:
                print(f"  WARNING: could not hash {f}: {e}")
            if i % 2000 == 0:
                print(f"  ... {i:,}/{len(files):,}")
    return records


def find_near_duplicates(records, threshold):
    """
    O(n^2) pairwise Hamming distance comparison.
    Fine for tens of thousands of images; for very large pools, bucket by
    hash prefix first to prune comparisons.
    """
    flagged = []
    n = len(records)
    print(f"\nComparing {n:,} images pairwise (threshold={threshold}) ...")
    for i in range(n):
        pool_a, fname_a, hash_a = records[i]
        for j in range(i + 1, n):
            pool_b, fname_b, hash_b = records[j]
            dist = hash_a - hash_b  # Hamming distance
            if dist <= threshold:
                flagged.append({
                    "pool_a": pool_a, "file_a": fname_a,
                    "pool_b": pool_b, "file_b": fname_b,
                    "hamming_distance": dist,
                    "cross_split": pool_a != pool_b,
                })
        if i and i % 1000 == 0:
            print(f"  ... compared {i:,}/{n:,}")
    return flagged


def main():
    ap = argparse.ArgumentParser(description="Near-duplicate detector (pHash) across image pools")
    ap.add_argument(
        "--pool", action="append", required=True,
        help="name=/path/to/dir, repeatable. e.g. --pool dff_train=/data/DFF/Split/train/Real",
    )
    ap.add_argument("--threshold", type=int, default=5, help="Max Hamming distance to flag as duplicate")
    ap.add_argument("--output", default="duplicate_report.csv", help="Output CSV path")
    args = ap.parse_args()

    pools = dict(parse_pool_arg(p) for p in args.pool)

    records = compute_hashes(pools)
    flagged = find_near_duplicates(records, args.threshold)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["pool_a", "file_a", "pool_b", "file_b", "hamming_distance", "cross_split"]
        )
        w.writeheader()
        w.writerows(flagged)

    cross_split_leaks = [r for r in flagged if r["cross_split"]]

    print("\n===================== SUMMARY =====================")
    print(f"Total images hashed         : {len(records):,}")
    print(f"Near-duplicate pairs found  : {len(flagged):,}")
    print(f"  - within the same pool    : {len(flagged) - len(cross_split_leaks):,}")
    print(f"  - ACROSS pools (LEAKAGE)  : {len(cross_split_leaks):,}")
    print(f"Report written to           : {args.output}")
    if cross_split_leaks:
        print("\nWARNING: cross-split near-duplicates found. These are candidate leakage")
        print("pairs and should be resolved (e.g. remove from train, keep in test) before")
        print("reporting results. See the report CSV for full details.")
    print("=====================================================")


if __name__ == "__main__":
    main()
