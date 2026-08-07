import os
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm
import torch

# ==========================================================
# SETTINGS
# ==========================================================

ROOT = os.path.expanduser("~/Dataset/DFF")

REAL_DIR = os.path.join(ROOT, "Real")
INPAINT_DIR = os.path.join(ROOT, "Fake", "inpainting")
TEXT2IMG_DIR = os.path.join(ROOT, "Fake", "text2img")

OUTPUT_ROOT = os.path.expanduser("~/Dataset/DFF_Cropped")

OUT_REAL = os.path.join(OUTPUT_ROOT, "Real")
OUT_INPAINT = os.path.join(OUTPUT_ROOT, "Fake", "inpainting")
OUT_TEXT2IMG = os.path.join(OUTPUT_ROOT, "Fake", "text2img")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

IMAGE_SIZE = 224
MARGIN = 20  # extra pixels around face

# ==========================================================
# CREATE OUTPUT FOLDERS
# ==========================================================

os.makedirs(OUT_REAL, exist_ok=True)
os.makedirs(OUT_INPAINT, exist_ok=True)
os.makedirs(OUT_TEXT2IMG, exist_ok=True)

# ==========================================================
# DEVICE
# ==========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ==========================================================
# MTCNN
# ==========================================================

mtcnn = MTCNN(
    image_size=224,
    margin=20,
    keep_all=False,   # Only the largest face
    post_process=True,
    device=device
)

# ==========================================================
# READ FILENAMES
# ==========================================================

def get_image_set(folder):
    return {
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    }

real_set = get_image_set(REAL_DIR)
inpaint_set = get_image_set(INPAINT_DIR)
text2img_set = get_image_set(TEXT2IMG_DIR)

# ==========================================================
# FIND COMMON FILES
# ==========================================================

common_files = sorted(real_set & inpaint_set & text2img_set)

missing_real = sorted((inpaint_set | text2img_set) - real_set)
missing_inpaint = sorted((real_set | text2img_set) - inpaint_set)
missing_text2img = sorted((real_set | inpaint_set) - text2img_set)

# Save reports
with open(os.path.join(OUTPUT_ROOT, "missing_real.txt"), "w") as f:
    f.write("\n".join(missing_real))

with open(os.path.join(OUTPUT_ROOT, "missing_inpainting.txt"), "w") as f:
    f.write("\n".join(missing_inpaint))

with open(os.path.join(OUTPUT_ROOT, "missing_text2img.txt"), "w") as f:
    f.write("\n".join(missing_text2img))

print("\n" + "=" * 70)
print(f"Real images       : {len(real_set):,}")
print(f"Inpainting images : {len(inpaint_set):,}")
print(f"Text2img images   : {len(text2img_set):,}")
print("-" * 70)
print(f"Common images     : {len(common_files):,}")
print(f"Missing in Real         : {len(missing_real):,}")
print(f"Missing in Inpainting   : {len(missing_inpaint):,}")
print(f"Missing in Text2img     : {len(missing_text2img):,}")
print("=" * 70)

# ==========================================================
# CROP FUNCTION
# ==========================================================

# ==========================================================
# CROP FUNCTION
# ==========================================================

def crop_and_save(src_path, dst_path):
    try:
        img = Image.open(src_path).convert("RGB")

        # Detect, crop, resize and return the largest face
        face = mtcnn(img)

        # No face detected
        if face is None:
            return False

        # Convert tensor to PIL image
        face = face.permute(1, 2, 0).cpu().numpy()
        face = ((face + 1) / 2 * 255).clip(0, 255).astype("uint8")

        face = Image.fromarray(face).convert("RGB")

        # Save as JPEG
        face.save(dst_path, "JPEG", quality=95)

        return True

    except Exception as e:
        print(f"\nERROR: {src_path}")
        print(e)
        return False
# ==========================================================
# PROCESS DATASET
# ==========================================================

failed_files = []
processed = 0

for filename in tqdm(common_files, desc="Cropping images"):

    real_src = os.path.join(REAL_DIR, filename)
    inp_src = os.path.join(INPAINT_DIR, filename)
    txt_src = os.path.join(TEXT2IMG_DIR, filename)

    real_dst = os.path.join(OUT_REAL, filename)
    inp_dst = os.path.join(OUT_INPAINT, filename)
    txt_dst = os.path.join(OUT_TEXT2IMG, filename)

    ok1 = crop_and_save(real_src, real_dst)
    ok2 = crop_and_save(inp_src, inp_dst)
    ok3 = crop_and_save(txt_src, txt_dst)

    # Keep only complete triplets
    if ok1 and ok2 and ok3:
        processed += 1
    else:
        failed_files.append(filename)

        # Remove partial outputs if any
        for p in [real_dst, inp_dst, txt_dst]:
            if os.path.exists(p):
                os.remove(p)

# ==========================================================
# SAVE FAILURE REPORT
# ==========================================================

with open(os.path.join(OUTPUT_ROOT, "mtcnn_failed.txt"), "w") as f:
    f.write("\n".join(failed_files))

# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"Common image triplets : {len(common_files):,}")
print(f"Successfully cropped  : {processed:,}")
print(f"Face detection failed : {len(failed_files):,}")
print("=" * 70)
print(f"Cropped dataset saved to: {OUTPUT_ROOT}")
print("Done!")
