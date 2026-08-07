#!/usr/bin/env python3
"""
identity_split_pipeline.py
===========================
One-shot, reproducible, identity-safe train/val/test split for DFF-style
Real/Fake datasets.

WHAT THIS DOES
--------------
1. Loads real_identity_metadata.csv (filename -> identity -> source).
2. Produces identity_statistics.csv + a console distribution report.
3. Splits UNIQUE IDENTITIES (not images) into train/val/test using a fixed
   random seed, so every image of a given identity always lands in the
   same split (zero identity leakage for Real).
4. Verifies train/val/test identity sets are pairwise disjoint.
5. Splits Fake images by following their corresponding Real image's split:
   each Fake image is matched to a Real image by identical filename, and
   is placed in whichever split (train/val/test) that Real image landed
   in. Real images with no matching Fake file, and Fake files with no
   matching Real filename, are reported separately.
6. Writes train.csv / validation.csv / test.csv (Real) and
   train_fake.csv / validation_fake.csv / test_fake.csv (Fake).
7. Physically copies files into:
       Split/train/Real, Split/train/Fake
       Split/validation/Real, Split/validation/Fake
       Split/test/Real, Split/test/Fake
8. Prints a full final report (counts, identities, leakage check, seed).

USAGE
-----
python identity_split_pipeline.py \
    --root /path/to/DFFD_A_1/Original \
    --metadata /path/to/DFFD_A_1/Original/real_identity_metadata.csv \
    --output /path/to/DFFD_A_1/Original/Split \
    --seed 42 --train 0.80 --val 0.05 --test 0.15

Add --dry-run to only generate the CSVs/report without copying files
(useful to sanity-check the split before touching disk).

The script auto-detects common column-name variants in the metadata CSV
(filename/file/image/image_name, identity/name/person, source/dataset).
If it can't find a required column it stops and tells you exactly what
columns it saw, instead of guessing.
"""

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict, Counter
from pathlib import Path


# --------------------------------------------------------------------------- #
# Metadata loading
# --------------------------------------------------------------------------- #

def detect_column(fieldnames, candidates, label, required=True):
    lower_map = {f.strip().lower(): f for f in fieldnames}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    if required:
        sys.exit(
            f"ERROR: could not find a '{label}' column in the metadata CSV.\n"
            f"Columns found: {fieldnames}\n"
            f"Expected one of: {candidates}"
        )
    return None


def load_real_metadata(csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            sys.exit(f"ERROR: {csv_path} appears to be empty or has no header row.")

        fn_col = detect_column(
            reader.fieldnames,
            ["filename", "file", "image", "image_name", "file_name"],
            "filename",
        )
        id_col = detect_column(
            reader.fieldnames,
            ["identity", "name", "person", "person_id", "identity_name"],
            "identity",
        )
        src_col = detect_column(
            reader.fieldnames,
            ["source", "dataset", "origin"],
            "source",
            required=False,
        )

        rows = []
        for r in reader:
            filename = (r.get(fn_col) or "").strip()
            identity = (r.get(id_col) or "").strip()
            if not filename or not identity:
                continue
            source = (r.get(src_col) or "unknown").strip() if src_col else "unknown"
            rows.append({"filename": filename, "identity": identity, "source": source})

    if not rows:
        sys.exit(f"ERROR: no usable rows parsed from {csv_path}")
    return rows


# --------------------------------------------------------------------------- #
# Step 3: identity statistics
# --------------------------------------------------------------------------- #

def write_identity_statistics(rows, out_path: Path):
    by_identity = defaultdict(list)
    for r in rows:
        by_identity[r["identity"]].append(r)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Identity", "Images", "Source"])
        for identity, items in sorted(by_identity.items(), key=lambda kv: kv[0]):
            sources = Counter(it["source"] for it in items)
            main_source = sources.most_common(1)[0][0]
            w.writerow([identity, len(items), main_source])

    counts = [len(v) for v in by_identity.values()]
    dist = Counter(counts)

    print("\n=== Step 3: Identity statistics ===")
    print(f"Total identities        : {len(by_identity)}")
    print(f"Total images            : {sum(counts)}")
    for n_images in sorted(dist):
        print(f"Identities with {n_images} image(s) : {dist[n_images]}")
    print(f"Maximum images / identity: {max(counts)}")
    print(f"Average images / identity: {sum(counts) / len(counts):.3f}")
    print(f"Written -> {out_path}")

    return by_identity


# --------------------------------------------------------------------------- #
# Step 4: identity-level split
# --------------------------------------------------------------------------- #

def split_identities(identities, seed, train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        sys.exit(f"ERROR: train+val+test must sum to 1.0 (got {total})")

    ids = sorted(identities)          # deterministic order before shuffling
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    # test gets the remainder so counts always add up exactly to n
    n_test = n - n_train - n_val

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])

    assert len(train_ids) + len(val_ids) + len(test_ids) == n
    return train_ids, val_ids, test_ids


# --------------------------------------------------------------------------- #
# Step 5: leakage verification
# --------------------------------------------------------------------------- #

def verify_no_leakage(train_ids, val_ids, test_ids):
    overlaps = {
        "train ∩ val": train_ids & val_ids,
        "train ∩ test": train_ids & test_ids,
        "val ∩ test": val_ids & test_ids,
    }
    leaked = {k: v for k, v in overlaps.items() if v}
    if leaked:
        print("\n=== Step 5: Leakage verification ===")
        for k, v in leaked.items():
            print(f"ERROR: {k} overlap ({len(v)} identities): {sorted(v)[:10]}...")
        sys.exit("ABORTING: identity leakage detected.")
    print("\n=== Step 5: Leakage verification ===")
    print("Train ∩ Validation = ∅")
    print("Train ∩ Test       = ∅")
    print("Validation ∩ Test  = ∅")
    print("✓ Zero identity leakage")


# --------------------------------------------------------------------------- #
# Fake split (paired to Real by identical filename)
# --------------------------------------------------------------------------- #

def build_fake_splits_from_real(split_rows, fake_dir: Path):
    """
    For each split (train/validation/test), a Fake image is included if and
    only if a file with the exact same filename exists in fake_dir. This
    keeps every Real/Fake pair in the same split -- no independent random
    split or seed needed for Fake, since it simply mirrors Real.

    Returns:
        fake_splits: dict split_name -> list of fake filenames
        real_without_fake: dict split_name -> list of real filenames with no match
        orphan_fakes: list of fake filenames that don't match any Real filename
    """
    fake_files_on_disk = {p.name for p in fake_dir.iterdir() if p.is_file()}

    fake_splits = {}
    real_without_fake = {}
    matched_fake_files = set()

    for name, rows in split_rows.items():
        matched = []
        unmatched_real = []
        for r in rows:
            fn = r["filename"]
            if fn in fake_files_on_disk:
                matched.append(fn)
                matched_fake_files.add(fn)
            else:
                unmatched_real.append(fn)
        fake_splits[name] = sorted(matched)
        real_without_fake[name] = unmatched_real

    orphan_fakes = sorted(fake_files_on_disk - matched_fake_files)

    return fake_splits, real_without_fake, orphan_fakes


# --------------------------------------------------------------------------- #
# CSV writers + file copy
# --------------------------------------------------------------------------- #

def write_split_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "identity", "source"])
        for r in rows:
            w.writerow([r["filename"], r["identity"], r["source"]])


def write_fake_split_csv(path: Path, filenames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename"])
        for fn in filenames:
            w.writerow([fn])


def copy_files(src_dir: Path, filenames, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for fn in filenames:
        src = src_dir / fn
        if not src.exists():
            missing.append(fn)
            continue
        shutil.copy2(src, dest_dir / fn)
    return missing


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Identity-safe Real/Fake split pipeline")
    ap.add_argument("--root", required=True, help="Folder containing Real/ and Fake/")
    ap.add_argument("--metadata", required=True, help="Path to real_identity_metadata.csv")
    ap.add_argument("--output", required=True, help="Output folder for the Split/ tree and CSVs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--dry-run", action="store_true", help="Only write CSVs, skip copying files")
    args = ap.parse_args()

    root = Path(args.root)
    real_dir = root / "Real"
    fake_dir = root / "Fake"
    metadata_path = Path(args.metadata)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    for d in (real_dir, fake_dir):
        if not d.exists():
            sys.exit(f"ERROR: expected folder not found: {d}")

    # Step 1/2 already done upstream (rename + metadata creation).
    rows = load_real_metadata(metadata_path)

    # Step 3
    by_identity = write_identity_statistics(rows, out_root / "identity_statistics.csv")

    # Step 4
    print("\n=== Step 4: Identity-level split ===")
    train_ids, val_ids, test_ids = split_identities(
        by_identity.keys(), args.seed, args.train, args.val, args.test
    )
    print(f"Train identities      : {len(train_ids)}")
    print(f"Validation identities : {len(val_ids)}")
    print(f"Test identities       : {len(test_ids)}")

    # Step 5
    verify_no_leakage(train_ids, val_ids, test_ids)

    # Build per-split row lists (Real)
    split_rows = {"train": [], "validation": [], "test": []}
    for r in rows:
        ident = r["identity"]
        if ident in train_ids:
            split_rows["train"].append(r)
        elif ident in val_ids:
            split_rows["validation"].append(r)
        else:
            split_rows["test"].append(r)

    for name in ("train", "validation", "test"):
        write_split_csv(out_root / f"{name}.csv", split_rows[name])

    # Fake split: each Fake image follows its matching Real image's split
    # (matched by identical filename). No separate ratio/seed needed here.
    print("\n=== Fake image split (paired to Real by identical filename) ===")
    fake_splits, real_without_fake, orphan_fakes = build_fake_splits_from_real(split_rows, fake_dir)
    for name in ("train", "validation", "test"):
        write_fake_split_csv(out_root / f"{name}_fake.csv", fake_splits[name])
        print(f"{name:<10} fake images: {len(fake_splits[name])}"
              f" (unmatched real: {len(real_without_fake[name])})")
    if orphan_fakes:
        print(f"NOTE: {len(orphan_fakes)} Fake file(s) have no matching Real filename "
              f"and were excluded from all splits (e.g. {orphan_fakes[:5]}).")

    # Step 6: copy files
    missing_report = {}
    if not args.dry_run:
        print("\n=== Step 6: Copying files ===")
        for name in ("train", "validation", "test"):
            real_dest = out_root / "Split" / name / "Real"
            fake_dest = out_root / "Split" / name / "Fake"
            missing_real = copy_files(real_dir, [r["filename"] for r in split_rows[name]], real_dest)
            missing_fake = copy_files(fake_dir, fake_splits[name], fake_dest)
            if missing_real:
                missing_report[f"{name}/Real"] = missing_real
            if missing_fake:
                missing_report[f"{name}/Fake"] = missing_fake
            print(f"{name:<10} -> {real_dest} ({len(split_rows[name]) - len(missing_real)} copied), "
                  f"{fake_dest} ({len(fake_splits[name]) - len(missing_fake)} copied)")
    else:
        print("\n--dry-run set: skipped physical file copy.")

    # Final report
    print("\n===================== FINAL REPORT =====================")
    print(f"Random seed              : {args.seed}")
    print(f"Ratios (train/val/test)  : {args.train}/{args.val}/{args.test}  (identity split only)")
    print(f"Total Real images        : {len(rows)}")
    print(f"Total identities         : {len(by_identity)}")
    for name in ("train", "validation", "test"):
        print(f"{name.capitalize():<10} Real images : {len(split_rows[name]):>6}  "
              f"| identities: {len(train_ids) if name=='train' else len(val_ids) if name=='validation' else len(test_ids)}")
    for name in ("train", "validation", "test"):
        print(f"{name.capitalize():<10} Fake images : {len(fake_splits[name]):>6}  "
              f"(paired to Real by filename)")
    counts = [len(v) for v in by_identity.values()]
    print(f"Largest identity          : {max(counts)} images")
    print(f"Smallest identity         : {min(counts)} images")
    print(f"Average images/identity   : {sum(counts)/len(counts):.3f}")
    print("Leakage                  : none (verified in Step 5)")
    if orphan_fakes:
        print(f"Orphan Fake files         : {len(orphan_fakes)} (no matching Real filename, excluded)")
    if missing_report:
        print("\nWARNING: some files listed in metadata/splits were not found on disk:")
        for k, v in missing_report.items():
            print(f"  {k}: {len(v)} missing (e.g. {v[:5]})")
    print("==========================================================")


if __name__ == "__main__":
    main()