#!/usr/bin/env python3
"""
build_final_dataset.py

Purpose
-------
You have a folder (CroppedFaces) containing 286 identity folders, each named like:
    ID000072_Christian-Bale
Each identity folder contains two subfolders:
    Real/   -> some number of real images (1, 2, 4, ...)
    Fake/   -> some number of fake images (1, 2, ...)

This script:
    1. Scans all identity folders.
    2. Keeps only identities that have >=1 real image AND >=1 fake image.
    3. Randomly selects N identities (default 200) out of those valid ones.
    4. For each selected identity, randomly picks exactly ONE real image
       and ONE fake image.
    5. Copies them into a new output dataset:
           FinalDataset/Real/<IDENTITY_NAME>.<ext>
           FinalDataset/Fake/<IDENTITY_NAME>.<ext>
    6. Writes a CSV log (selection_log.csv) recording exactly which file was
       chosen for each identity, for auditing / reproducibility.

How to run
----------
1. Save this file anywhere, e.g. in your CroppedFaces folder or one level above.
2. Activate your conda env (you already have "deepfake" active):
       conda activate deepfake
3. Run it:
       python build_final_dataset.py \
           --source "/home/amna/GUI-APP/GUI-V9-Final-Dataset/DeepfakeAnnotationTool_v9_FinalDataset/CroppedFaces" \
           --output "/home/amna/GUI-APP/GUI-V9-Final-Dataset/DeepfakeAnnotationTool_v9_FinalDataset/FinalDataset" \
           --num-identities 200 \
           --seed 42

   If you just run it with no arguments, it defaults to:
       source = current directory (CroppedFaces)
       output = ./FinalDataset
       num-identities = 200
       seed = 42

4. Check the output:
       FinalDataset/Real/   -> 200 images
       FinalDataset/Fake/   -> 200 images
       FinalDataset/selection_log.csv -> record of what was picked

Notes
-----
- Uses a fixed random seed by default (42) so the selection is REPRODUCIBLE.
  Change --seed to get a different random selection.
- Matches "Real"/"real"/"REAL" and "Fake"/"fake"/"FAKE" folder name variants,
  in case your folders aren't perfectly consistent.
- Accepts .jpg, .jpeg, .png, .bmp, .webp image extensions.
- If fewer than --num-identities identities qualify (have both real & fake),
  the script will tell you how many are actually available and stop with
  a clear error rather than silently giving you fewer than you asked for.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_subfolder(identity_dir: Path, target_name: str) -> Path | None:
    """Find a subfolder matching target_name case-insensitively."""
    for child in identity_dir.iterdir():
        if child.is_dir() and child.name.lower() == target_name.lower():
            return child
    return None


def list_images(folder: Path) -> list[Path]:
    if folder is None or not folder.exists():
        return []
    return [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]


def main():
    parser = argparse.ArgumentParser(description="Build balanced Real/Fake final dataset.")
    parser.add_argument("--source", type=str, default=".",
                         help="Path to CroppedFaces folder containing identity subfolders.")
    parser.add_argument("--output", type=str, default="./FinalDataset",
                         help="Path to output folder to create (Real/ and Fake/ go inside it).")
    parser.add_argument("--num-identities", type=int, default=200,
                         help="How many identities to randomly select (default 200).")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for reproducibility (default 42).")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if not source.exists():
        raise SystemExit(f"ERROR: source folder does not exist: {source}")

    real_out = output / "Real"
    fake_out = output / "Fake"
    real_out.mkdir(parents=True, exist_ok=True)
    fake_out.mkdir(parents=True, exist_ok=True)

    # 1. Scan all identity folders
    identity_dirs = sorted([d for d in source.iterdir() if d.is_dir()])
    print(f"Found {len(identity_dirs)} identity folders in source.")

    valid_identities = []  # list of (identity_dir, real_images, fake_images)
    skipped = []

    for identity_dir in identity_dirs:
        real_folder = find_subfolder(identity_dir, "Real")
        fake_folder = find_subfolder(identity_dir, "Fake")

        real_images = list_images(real_folder) if real_folder else []
        fake_images = list_images(fake_folder) if fake_folder else []

        if len(real_images) >= 1 and len(fake_images) >= 1:
            valid_identities.append((identity_dir, real_images, fake_images))
        else:
            skipped.append((identity_dir.name, len(real_images), len(fake_images)))

    print(f"Identities with at least 1 real AND 1 fake image: {len(valid_identities)}")
    if skipped:
        print(f"Skipping {len(skipped)} identities missing real or fake images:")
        for name, nr, nf in skipped[:20]:
            print(f"   - {name}: real={nr}, fake={nf}")
        if len(skipped) > 20:
            print(f"   ... and {len(skipped) - 20} more")

    if len(valid_identities) < args.num_identities:
        raise SystemExit(
            f"ERROR: You asked for {args.num_identities} identities, but only "
            f"{len(valid_identities)} identities have both real and fake images. "
            f"Lower --num-identities or fix the missing folders."
        )

    # 2. Randomly select N identities
    random.seed(args.seed)
    selected = random.sample(valid_identities, args.num_identities)

    # 3. For each selected identity, randomly pick one real + one fake image, copy them
    log_rows = []
    for identity_dir, real_images, fake_images in selected:
        chosen_real = random.choice(real_images)
        chosen_fake = random.choice(fake_images)

        identity_name = identity_dir.name  # e.g. ID000072_Christian-Bale

        real_dest = real_out / f"{identity_name}{chosen_real.suffix.lower()}"
        fake_dest = fake_out / f"{identity_name}{chosen_fake.suffix.lower()}"

        shutil.copy2(chosen_real, real_dest)
        shutil.copy2(chosen_fake, fake_dest)

        log_rows.append({
            "identity": identity_name,
            "real_source": str(chosen_real),
            "real_dest": str(real_dest),
            "fake_source": str(chosen_fake),
            "fake_dest": str(fake_dest),
        })

    # 4. Write CSV log
    log_path = output / "selection_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["identity", "real_source", "real_dest", "fake_source", "fake_dest"])
        writer.writeheader()
        writer.writerows(log_rows)

    print("\nDONE.")
    print(f"Selected {len(selected)} identities.")
    print(f"Real images written to:  {real_out}  ({len(list(real_out.glob('*')))} files)")
    print(f"Fake images written to:  {fake_out}  ({len(list(fake_out.glob('*')))} files)")
    print(f"Selection log saved to:  {log_path}")


if __name__ == "__main__":
    main()
