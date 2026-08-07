import shutil
from pathlib import Path
import pandas as pd

# =====================================================
# Root directory
# =====================================================
ROOT = Path("/home/amna/Ready_to_send_Dataset/DFF/DFF_Without_Cropped")

REAL_DIR = ROOT / "Real"
INPAINT_DIR = ROOT / "Fake" / "inpainting"
TEXT2IMG_DIR = ROOT / "Fake" / "text2img"

CSV_FILES = [
    ROOT / "DFF_C_Train_Real_Selected_Metadata.csv",
    ROOT / "DFF_C_Val_Real_Selected_Metadata.csv",
    ROOT / "DFF_C_Test_Real_Selected_Meteadata.csv",
]

# =====================================================
# Destination folders
# =====================================================
DEST_REAL = ROOT / "Moved_DFFC" / "Real"
DEST_INPAINT = ROOT / "Moved_DFFC" / "Fake" / "inpainting"
DEST_TEXT2IMG = ROOT / "Moved_DFFC" / "Fake" / "text2img"

DEST_REAL.mkdir(parents=True, exist_ok=True)
DEST_INPAINT.mkdir(parents=True, exist_ok=True)
DEST_TEXT2IMG.mkdir(parents=True, exist_ok=True)

# =====================================================
# Read CSVs and build matching keys
# =====================================================
keys = set()

for csv_file in CSV_FILES:

    df = pd.read_csv(csv_file)

    for fname in df["filename"].astype(str):

        stem = Path(fname).stem

        parts = stem.split("_")

        # Keep only:
        # number_birthdate_year
        # Example:
        # 5831509_1918-05-17_2009_A._C._Lyles
        # ->
        # 5831509_1918-05-17_2009
        key = "_".join(parts[:3])

        keys.add(key)

print(f"Keys loaded: {len(keys)}")

# =====================================================
# Generic mover
# =====================================================
def move_matching(src_dir, dst_dir):

    moved = 0

    for img in src_dir.iterdir():

        if not img.is_file():
            continue

        stem = img.stem

        if stem in keys:
            shutil.move(str(img), str(dst_dir / img.name))
            moved += 1

    return moved

# =====================================================
# Move files
# =====================================================
real_count = move_matching(REAL_DIR, DEST_REAL)
inpaint_count = move_matching(INPAINT_DIR, DEST_INPAINT)
text2img_count = move_matching(TEXT2IMG_DIR, DEST_TEXT2IMG)

# =====================================================
# Summary
# =====================================================
print("\n========== SUMMARY ==========")
print(f"Moved Real images        : {real_count}")
print(f"Moved Inpainting images  : {inpaint_count}")
print(f"Moved Text2Img images    : {text2img_count}")
print("Done.")