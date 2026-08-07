import shutil
from pathlib import Path
from PIL import Image

# =====================================================
# Paths
# =====================================================

INPUT_DIR = Path("/home/amna/Ready_to_send_Dataset/DFF/DFF_Cropped_224/Fake/inpainting")

RGB_DIR = INPUT_DIR.parent / "inpainting_RGB"
GRAY_DIR = INPUT_DIR.parent / "inpainting_Grayscale"

RGB_DIR.mkdir(parents=True, exist_ok=True)
GRAY_DIR.mkdir(parents=True, exist_ok=True)

rgb_count = 0
gray_count = 0

# =====================================================
# Check each image
# =====================================================

for img_path in INPUT_DIR.iterdir():

    if not img_path.is_file():
        continue

    try:
        img = Image.open(img_path)

        # True grayscale image
        if img.mode in ("L", "1"):
            shutil.copy2(img_path, GRAY_DIR / img_path.name)
            gray_count += 1
            continue

        # RGB image
        if img.mode == "RGB":

            # Check if R=G=B everywhere
            r, g, b = img.split()

            if list(r.getdata()) == list(g.getdata()) == list(b.getdata()):
                shutil.copy2(img_path, GRAY_DIR / img_path.name)
                gray_count += 1
            else:
                shutil.copy2(img_path, RGB_DIR / img_path.name)
                rgb_count += 1

        else:
            # Convert unusual modes (RGBA, etc.) to RGB and test
            img = img.convert("RGB")

            r, g, b = img.split()

            if list(r.getdata()) == list(g.getdata()) == list(b.getdata()):
                shutil.copy2(img_path, GRAY_DIR / img_path.name)
                gray_count += 1
            else:
                shutil.copy2(img_path, RGB_DIR / img_path.name)
                rgb_count += 1

    except Exception as e:
        print(f"Error: {img_path.name} -> {e}")

print("=" * 50)
print(f"RGB images       : {rgb_count}")
print(f"Grayscale images : {gray_count}")
print("=" * 50)
