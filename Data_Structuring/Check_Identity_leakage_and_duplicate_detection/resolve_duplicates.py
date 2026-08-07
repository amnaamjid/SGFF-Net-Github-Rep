#!/usr/bin/env python3
"""
resolve_duplicates.py
========================
Reads duplicate_report.csv and decides, per pair, which file to remove using
a priority rule -- never removes from a protected/evaluation pool, only from
the "lower priority" (trained-on) side. Pairs between two equally-protected
pools (e.g. two eval-only sets) are NOT removed, just logged for your records
-- that's not leakage, since neither side is training data.

PRIORITY (1 = most protected, never removed from)
---------------------------------------------------
  1: diffface, dffd_b, dff_c_test, dffd_a_test
  2: dff_c_val, dffd_a_val
  3: dff_c_train, dffd_a_train

For any duplicate pair, the file belonging to the HIGHER number (less
protected) pool is the one removed. Equal priority on both sides -> logged
to needs_manual_review.csv, nothing removed automatically.

SAFETY: by default this is a DRY RUN. It only prints/writes what WOULD be
removed. Add --apply to actually move the flagged files into a quarantine
folder (not permanently delete -- so you can undo if needed).

USAGE
-----
pip install pandas --break-system-packages

# 1. Dry run first -- just see what would happen
python resolve_duplicates.py \
    --report duplicate_report.csv \
    --pool dff_c_train=/home/amna/Ready_to_send_Dataset/DFF/This_is_final_DFF_C/Split_output/Split/train/Real \
    --pool dff_c_val=/home/amna/Ready_to_send_Dataset/DFF/This_is_final_DFF_C/Split_output/Split/validation/Real \
    --pool dff_c_test=/home/amna/Ready_to_send_Dataset/DFF/This_is_final_DFF_C/Split_output/Split/test/Real \
    --pool dffd_a_train=/home/amna/Ready_to_send_Dataset/DFFD/DFFDA/This_is_final_DFFD_A_dataset_balanced/train/Real \
    --pool dffd_a_val=/home/amna/Ready_to_send_Dataset/DFFD/DFFDA/This_is_final_DFFD_A_dataset_balanced/val/Real \
    --pool dffd_a_test=/home/amna/Ready_to_send_Dataset/DFFD/DFFDA/This_is_final_DFFD_A_dataset_balanced/Test/Real \
    --pool dffd_b=/home/amna/Ready_to_send_Dataset/DFFD/DFFDB/This_is_final_DFFD_B_eval_balanced/Real \
    --pool diffface=/home/amna/Ready_to_send_Dataset/DiffFacee/This_is_final_DiffFace/Real \
    --quarantine /home/amna/Ready_to_send_Dataset/duplicates_removed

# 2. Once you've reviewed removal_list_*.csv and are happy, re-run with --apply
#    (same command, just add --apply) to actually move the files.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("Missing dependency. Install with:\n  pip install pandas --break-system-packages")

PRIORITY = {
    "diffface": 1,
    "dffd_b": 1,
    "dff_c_test": 1,
    "dffd_a_test": 1,
    "dff_c_val": 2,
    "dffd_a_val": 2,
    "dff_c_train": 3,
    "dffd_a_train": 3,
}

# ONLY these pool pairs are checked for duplicate removal. Any pair NOT in
# this set is IGNORED completely, even if it appears in duplicate_report.csv
# -- e.g. dffd_a_train vs diffface is ignored, because DiffFace is only ever
# used to evaluate a model trained on DFF_C, never a model trained on DFFD_A.
# Overlap between two pools that are never used together in the same
# experiment cannot leak into anything, so it doesn't matter.
RELEVANT_PAIRS = {
    frozenset({"dff_c_train", "diffface"}),
    frozenset({"dff_c_val", "diffface"}),
    frozenset({"dff_c_train", "dffd_a_train"}),
    frozenset({"dff_c_train", "dffd_a_val"}),
    frozenset({"dff_c_train", "dffd_a_test"}),
    frozenset({"dff_c_val", "dffd_a_train"}),
    frozenset({"dff_c_val", "dffd_a_val"}),
    frozenset({"dff_c_val", "dffd_a_test"}),
    frozenset({"dff_c_train", "dffd_b"}),
    frozenset({"dff_c_val", "dffd_b"}),
    frozenset({"dffd_a_train", "dffd_b"}),
    frozenset({"dffd_a_val", "dffd_b"}),
    frozenset({"dff_c_train", "dff_c_val"}),
    frozenset({"dff_c_train", "dff_c_test"}),
    frozenset({"dff_c_val", "dff_c_test"}),
    frozenset({"dffd_a_train", "dffd_a_val"}),
    frozenset({"dffd_a_train", "dffd_a_test"}),
    frozenset({"dffd_a_val", "dffd_a_test"}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--pool", action="append", required=True, help="name=path/to/Real/folder, repeatable")
    ap.add_argument("--quarantine", required=True, type=Path, help="Where flagged files get moved to")
    ap.add_argument("--apply", action="store_true", help="Actually move files. Without this: dry run only.")
    args = ap.parse_args()

    pools = {}
    for spec in args.pool:
        name, path = spec.split("=", 1)
        pools[name] = Path(path)
        if name not in PRIORITY:
            sys.exit(f"ERROR: pool '{name}' has no priority assigned. Edit the PRIORITY dict.")

    df = pd.read_csv(args.report)
    cross = df[df["cross_split"] == True].copy()  # noqa: E712
    print(f"Cross-pool pairs to resolve: {len(cross):,}\n")

    # to_remove[pool_name] = set of filenames
    to_remove = {name: set() for name in pools}
    manual_review = []

    for row in cross.itertuples():
        pa, pb = row.pool_a, row.pool_b
        if frozenset({pa, pb}) not in RELEVANT_PAIRS:
            continue  # these two pools are never used together in any experiment -- ignore
        prio_a, prio_b = PRIORITY[pa], PRIORITY[pb]
        if prio_a == prio_b:
            manual_review.append(row)
            continue
        # remove from whichever has the HIGHER priority number (less protected)
        if prio_a > prio_b:
            to_remove[pa].add(row.file_a)
        else:
            to_remove[pb].add(row.file_b)

    print("=" * 60)
    print("REMOVAL PLAN")
    print("=" * 60)
    total_remove = 0
    for pool_name, files in to_remove.items():
        if files:
            print(f"  {pool_name:<14}: {len(files):,} file(s) to remove")
            total_remove += len(files)
    print(f"\nTotal files flagged for removal: {total_remove:,}")
    print(f"Logged for manual review (both sides equally protected): {len(manual_review):,}")

    # write removal lists
    out_dir = Path("duplicate_resolution")
    out_dir.mkdir(exist_ok=True)
    for pool_name, files in to_remove.items():
        if not files:
            continue
        path = out_dir / f"removal_list_{pool_name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename"])
            for fn in sorted(files):
                w.writerow([fn])
        print(f"  -> wrote {path}")

    if manual_review:
        path = out_dir / "needs_manual_review.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pool_a", "file_a", "pool_b", "file_b", "hamming_distance"])
            for r in manual_review:
                w.writerow([r.pool_a, r.file_a, r.pool_b, r.file_b, r.hamming_distance])
        print(f"  -> wrote {path} (both sides eval-only, informational, no action needed for leakage purposes)")

    if not args.apply:
        print("\nDRY RUN -- no files were moved. Review the CSVs in duplicate_resolution/,")
        print("then re-run this exact command with --apply to actually quarantine the files.")
        return

    print("\nApplying: moving flagged files to quarantine (not deleting)...")
    args.quarantine.mkdir(parents=True, exist_ok=True)
    moved, missing = 0, 0
    for pool_name, files in to_remove.items():
        if not files:
            continue
        src_dir = pools[pool_name]
        dest_dir = args.quarantine / pool_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for fn in files:
            src = src_dir / fn
            if not src.exists():
                print(f"  WARNING: {src} not found, skipping")
                missing += 1
                continue
            shutil.move(str(src), str(dest_dir / fn))
            moved += 1

    print(f"\nMoved {moved:,} files to {args.quarantine}")
    if missing:
        print(f"WARNING: {missing} files were already missing (already removed / renamed?)")
    print("\nIMPORTANT: your Real/Fake pairing means each removed Real image likely has a")
    print("matching Fake image with the same filename in the sibling Fake/ folder. Check")
    print("those Fake folders too and remove the matching files so Real/Fake counts stay")
    print("paired -- this script only touched the Real pools you passed in via --pool.")


if __name__ == "__main__":
    main()