#!/usr/bin/env python3

import shutil
from pathlib import Path

# =====================================================
# Paths
# =====================================================

SOURCE_DIRS = [
    Path.home() / "HPC_DATA/DFF/DFF_A/Origianls/Real",
    Path.home() / "HPC_DATA/DFF/DFF_B/Original/Real",
]

DEST_DIR = Path.home() / "HPC_DATA/DFF/Real"

# =====================================================

DEST_DIR.mkdir(parents=True, exist_ok=True)

copied = 0
skipped = 0

valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

for src_dir in SOURCE_DIRS:

    print(f"\nProcessing: {src_dir}")

    if not src_dir.exists():
        print("  Folder not found. Skipping.")
        continue

    for file in sorted(src_dir.iterdir()):

        if not file.is_file():
            continue

        if file.suffix.lower() not in valid_ext:
            continue

        dest_file = DEST_DIR / file.name

        if dest_file.exists():
            skipped += 1
            continue

        shutil.copy2(file, dest_file)
        copied += 1

print("\n===================================")
print("Finished")
print("===================================")
print(f"Copied : {copied}")
print(f"Skipped: {skipped}")
print(f"Total in destination: {len(list(DEST_DIR.glob('*')))}")
