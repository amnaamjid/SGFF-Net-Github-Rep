from pathlib import Path
import pandas as pd

# ====================================================
# Folder containing the real images
# ====================================================
image_dir = Path("/home/amna/Ready_to_send_Dataset/DFFD/Celeb-HQ-Real-Transforamtion")

output_excel = image_dir.parent / "Celeb-HQ-Real-Transforamtion_Real_Selected_Metadata.xlsx"

valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

records = []

for img in sorted(image_dir.iterdir()):

    if not img.is_file():
        continue

    if img.suffix.lower() not in valid_ext:
        continue

    stem = img.stem            # e.g. 19279_Carlos_Tevez
    ext = img.suffix.lower()   # .jpg

    # Split only at the first underscore
    parts = stem.split("_", 1)

    if len(parts) == 2:
        hq_index = parts[0]
        identity = parts[1].replace("_", " ")
    else:
        hq_index = ""
        identity = ""

    records.append({
        "HQ_Index": int(hq_index) if hq_index.isdigit() else hq_index,
        "Identity_Name": identity,
        "Image_File": img.name,
        "Image_Extension": ext
    })

# Create DataFrame
df = pd.DataFrame(records)

# Sort by HQ index
df = df.sort_values("HQ_Index").reset_index(drop=True)

# Save Excel
df.to_excel(output_excel, index=False)

print("=" * 60)
print(f"Total images : {len(df)}")
print(f"Saved to     : {output_excel}")
print("=" * 60)