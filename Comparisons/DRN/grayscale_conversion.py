import os
from PIL import Image

# ==========================================================
# Paths
# ==========================================================

base_path = "/home/amna/projects/Comparison/This_is_final_DFF_C"

input_folders = ["Train", "Val", "Test"]
classes = ["Real", "Fake"]

output_root = os.path.join(base_path, "Grayscale")

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# ==========================================================
# Convert to Grayscale
# ==========================================================

for folder in input_folders:

    for cls in classes:

        input_dir = os.path.join(base_path, folder, cls)
        output_dir = os.path.join(output_root, folder, cls)

        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):

            if not filename.lower().endswith(valid_extensions):
                continue

            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                img = Image.open(input_path).convert("L")   # Convert to grayscale
                img.save(output_path)

            except Exception as e:
                print(f"Error: {input_path}")
                print(e)

        print(f"Finished: {folder}/{cls}")

print("\nAll images converted successfully!")