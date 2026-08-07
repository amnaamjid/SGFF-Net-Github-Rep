import shutil
from pathlib import Path

# =====================================================
# Paths
# =====================================================

ROOT = Path("/home/amna/Ready_to_send_Dataset/DFF/DFF_Without_Cropped/DFF_Cropped_224")

REAL_DIR = ROOT / "Real_RGB"
FAKE_DIR = ROOT / "text2img_RGB"

OUT_REAL = ROOT / "DFF_Text2Img_RGB" / "Real"
OUT_FAKE = ROOT / "DFF_Text2Img_RGB" / "Fake"

OUT_REAL.mkdir(parents=True, exist_ok=True)
OUT_FAKE.mkdir(parents=True, exist_ok=True)

# =====================================================
# Copy paired images
# =====================================================

copied = 0
missing = 0
missing_files = []

for fake_img in sorted(FAKE_DIR.iterdir()):

    if not fake_img.is_file():
        continue

    real_img = REAL_DIR / fake_img.name

    if real_img.exists():

        shutil.copy2(real_img, OUT_REAL / real_img.name)
        shutil.copy2(fake_img, OUT_FAKE / fake_img.name)

        copied += 1

    else:

        missing += 1
        missing_files.append(fake_img.name)

# =====================================================
# Save missing list
# =====================================================

with open(ROOT / "missing_real_images.txt", "w") as f:
    for name in missing_files:
        f.write(name + "\n")

# =====================================================
# Summary
# =====================================================

print("=" * 50)
print(f"Fake images scanned : {copied + missing}")
print(f"Pairs copied        : {copied}")
print(f"Missing real images : {missing}")
print("=" * 50)

print("Dataset created:")
print(ROOT / "DFF_Text2Img_RGB")
