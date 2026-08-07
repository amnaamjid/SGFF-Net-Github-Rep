#!/usr/bin/env python3

import random
import shutil
from pathlib import Path

# ==========================================================
# Configuration
# ==========================================================

SEED = 42
random.seed(SEED)

ROOT = Path.home() / "HPC_DATA" / "DFF"

REAL_DIR = ROOT / "Real"

FAKE_A_DIR = ROOT / "DFF_A" / "Origianls" / "Fake"
FAKE_B_DIR = ROOT / "DFF_B" / "Original" / "Fake"

OUTPUT_DIR = ROOT / "Fake"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXT = {".jpg", ".jpeg", ".png"}

# ==========================================================
# Read all real filenames
# ==========================================================

real_files = sorted([
    f.name
    for f in REAL_DIR.iterdir()
    if f.is_file() and f.suffix.lower() in VALID_EXT
])

total = len(real_files)

print(f"Total real images : {total}")

# Shuffle reproducibly
random.shuffle(real_files)

half = total // 2

files_for_A = set(real_files[:half])
files_for_B = set(real_files[half:])

print(f"Assigned to DFF_A : {len(files_for_A)}")
print(f"Assigned to DFF_B : {len(files_for_B)}")

# ==========================================================
# Copy files
# ==========================================================

copied_A = 0
copied_B = 0
fallback_A = 0
fallback_B = 0
missing = []

for filename in real_files:

    src_A = FAKE_A_DIR / filename
    src_B = FAKE_B_DIR / filename
    dst = OUTPUT_DIR / filename

    # --------------------------
    # Assigned to DFF_A
    # --------------------------
    if filename in files_for_A:

        if src_A.exists():
            shutil.copy2(src_A, dst)
            copied_A += 1

        elif src_B.exists():
            shutil.copy2(src_B, dst)
            copied_B += 1
            fallback_B += 1

        else:
            missing.append(filename)

    # --------------------------
    # Assigned to DFF_B
    # --------------------------
    else:

        if src_B.exists():
            shutil.copy2(src_B, dst)
            copied_B += 1

        elif src_A.exists():
            shutil.copy2(src_A, dst)
            copied_A += 1
            fallback_A += 1

        else:
            missing.append(filename)

# ==========================================================
# Save missing report
# ==========================================================

report = OUTPUT_DIR / "missing_fake_images.txt"

with open(report, "w") as f:
    for name in missing:
        f.write(name + "\n")

# ==========================================================
# Summary
# ==========================================================

print("\n" + "=" * 60)
print("Finished")
print("=" * 60)

print(f"Copied from DFF_A        : {copied_A}")
print(f"Copied from DFF_B        : {copied_B}")

print(f"Fallback A -> B          : {fallback_B}")
print(f"Fallback B -> A          : {fallback_A}")

print(f"Missing fake images      : {len(missing)}")

print(f"Total fake images copied : {copied_A + copied_B}")

print(f"\nOutput folder:")
print(OUTPUT_DIR)

print(f"\nMissing report:")
print(report)
