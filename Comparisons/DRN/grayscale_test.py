import os
from PIL import Image

# ==========================================
# PATHS
# ==========================================

input_root = "/home/amna/projects/Comparison/DRN/Test"
output_root = "/home/amna/projects/Comparison/DRN/Grayscale"

datasets = [
    "DFFD_Test",
    "DFF_C_Test",
    "DiffFace_A_Test"
]

valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ==========================================
# CONVERT
# ==========================================

for dataset in datasets:

    print(f"\nProcessing {dataset}")

    input_dataset = os.path.join(input_root, dataset)
    output_dataset = os.path.join(output_root, dataset)

    if not os.path.exists(input_dataset):
        print(f"Dataset not found: {input_dataset}")
        continue

    for cls in ["Real", "Fake"]:

        input_class = os.path.join(input_dataset, cls)
        output_class = os.path.join(output_dataset, cls)

        if not os.path.exists(input_class):
            print(f"Missing folder: {input_class}")
            continue

        os.makedirs(output_class, exist_ok=True)

        count = 0

        for filename in os.listdir(input_class):

            if not filename.lower().endswith(valid_ext):
                continue

            input_path = os.path.join(input_class, filename)
            output_path = os.path.join(output_class, filename)

            try:
                img = Image.open(input_path).convert("L")
                img.save(output_path)
                count += 1
            except Exception as e:
                print(f"Failed: {input_path}")
                print(e)

        print(f"{cls}: {count} images converted.")

print("\nDone!")
print(f"Grayscale datasets saved to:\n{output_root}")
