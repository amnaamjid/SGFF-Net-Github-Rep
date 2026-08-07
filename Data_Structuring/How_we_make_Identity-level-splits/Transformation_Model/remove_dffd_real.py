import shutil
from pathlib import Path
import pandas as pd

# =====================================================
# Paths
# =====================================================

ROOT = Path("/home/amna/Ready_to_send_Dataset/DFFD")

REAL_DIR = ROOT / "Celeb-HQ-Real"

REMOVED_DIR = ROOT / "Removed_CelebHQ"

REMOVED_DIR.mkdir(exist_ok=True)

# =====================================================
# Excel files
# =====================================================

EXCEL_FILES = [
    ROOT / "DFFD_A_Train_Real_Selected_Metadata.xlsx",
    ROOT / "DFFD_A_Val_Real_Selected_Metadata.xlsx",
    ROOT / "DFFD_A_Test_Real_Selected_Metadata.xlsx",
    ROOT / "DFFD_B_Real_Selected_Metadata.xlsx",
    ROOT / "DiffFace_Real_Selected_Metadata.xlsx",
]

# =====================================================
# Read filenames to remove
# =====================================================

remove_files = set()

for excel in EXCEL_FILES:

    print(f"Reading {excel.name}")

    df = pd.read_excel(excel)

    remove_files.update(df["filename"].astype(str))

print(f"\nUnique filenames to remove: {len(remove_files)}")

# =====================================================
# Move images
# =====================================================

moved = 0
missing = 0
missing_files = []

for filename in sorted(remove_files):

    src = REAL_DIR / filename

    if src.exists():

        shutil.move(str(src), str(REMOVED_DIR / filename))
        moved += 1

    else:

        missing += 1
        missing_files.append(filename)

# =====================================================
# Save missing list
# =====================================================

pd.DataFrame({
    "filename": missing_files
}).to_excel(ROOT / "missing_celebhq_images.xlsx", index=False)

# =====================================================
# Summary
# =====================================================

print("\n==============================")
print(f"Moved images   : {moved}")
print(f"Missing images : {missing}")
print(f"Remaining      : {len(list(REAL_DIR.glob('*')))}")
print("==============================")
