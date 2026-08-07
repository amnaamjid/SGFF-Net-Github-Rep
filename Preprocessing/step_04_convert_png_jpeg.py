import os
from PIL import Image

# Root folder
ROOT_DIR = "/home/amna/Dataset/DFFD/Fake/pggan_v2"

converted = 0
failed = 0

for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if file.lower().endswith(".png"):
            png_path = os.path.join(root, file)
            jpg_path = os.path.splitext(png_path)[0] + ".jpg"

            try:
                # Open image
                img = Image.open(png_path)

                # Convert to RGB (required for JPEG)
                img = img.convert("RGB")

                # Resize
                img = img.resize((224, 224), Image.LANCZOS)

                # Save as JPEG
                img.save(jpg_path, "JPEG", quality=95)

                # Delete original PNG
                os.remove(png_path)

                converted += 1
                print(f"[OK] {png_path}")

            except Exception as e:
                failed += 1
                print(f"[FAILED] {png_path}")
                print(e)

print("\n==============================")
print(f"Converted : {converted}")
print(f"Failed    : {failed}")
print("==============================")
