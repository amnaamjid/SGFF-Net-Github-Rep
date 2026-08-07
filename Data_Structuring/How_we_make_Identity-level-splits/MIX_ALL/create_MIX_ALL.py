import os
import shutil
from pathlib import Path

# =====================================================
# INPUT DATASETS
# =====================================================

DFFC = Path("/home/amna/HPC_DATA/DFF/DFF_C/80-5-15/Split_output/Split")
DFFDA = Path("/home/amna/HPC_DATA/DFFD/DFFD_A_dataset_balanced")

# =====================================================
# OUTPUT
# =====================================================

OUTPUT = Path("/home/amna/HPC_DATA/Mixed_DFFC_DFFDA")

# =====================================================
# Folder names
# =====================================================

SPLITS = {
    "train": "train",
    "validation": "val",
    "test": "test"
}

PREFIX = {
    "DFFC": "DFFC",
    "DFFDA": "DFFDA"
}


def copy_images(src_dir, dst_dir, prefix):
    """
    Copy all images while prefixing filenames.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for img in sorted(src_dir.iterdir()):

        if not img.is_file():
            continue

        if img.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            continue

        new_name = f"{prefix}_{img.name}"

        shutil.copy2(img, dst_dir / new_name)

        count += 1

    return count


total = {
    "train_real":0,
    "train_fake":0,
    "val_real":0,
    "val_fake":0,
    "test_real":0,
    "test_fake":0
}

for dffc_split, dffda_split in SPLITS.items():

    print("="*60)
    print(dffc_split)

    out_real = OUTPUT / dffc_split / "Real"
    out_fake = OUTPUT / dffc_split / "Fake"

    # ---------------- DFF_C ----------------

    r = copy_images(
        DFFC / dffc_split / "Real",
        out_real,
        PREFIX["DFFC"]
    )

    f = copy_images(
        DFFC / dffc_split / "Fake",
        out_fake,
        PREFIX["DFFC"]
    )

    print(f"DFF_C   Real : {r}")
    print(f"DFF_C   Fake : {f}")

    # ---------------- DFFD_A ----------------

    r2 = copy_images(
        DFFDA / dffda_split / "Real",
        out_real,
        PREFIX["DFFDA"]
    )

    f2 = copy_images(
        DFFDA / dffda_split / "Fake",
        out_fake,
        PREFIX["DFFDA"]
    )

    print(f"DFFD_A  Real : {r2}")
    print(f"DFFD_A  Fake : {f2}")

    total[f"{'val' if dffc_split=='validation' else dffc_split}_real"] = r+r2
    total[f"{'val' if dffc_split=='validation' else dffc_split}_fake"] = f+f2

print("\n")
print("="*60)
print("FINAL DATASET")
print("="*60)

print(f"Train Real : {total['train_real']}")
print(f"Train Fake : {total['train_fake']}")
print()

print(f"Val Real   : {total['val_real']}")
print(f"Val Fake   : {total['val_fake']}")
print()

print(f"Test Real  : {total['test_real']}")
print(f"Test Fake  : {total['test_fake']}")
print()

print("Output:")
print(OUTPUT)
