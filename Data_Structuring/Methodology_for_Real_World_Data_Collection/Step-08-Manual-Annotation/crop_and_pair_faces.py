"""
crop_and_pair_faces.py
Post-processing step for FinalDataset/ (the output of build_final_dataset.py):

  1. Detects and crops the face in every image using MTCNN. Images where
     no sufficiently confident, reasonably-sized face is found are
     discarded (not copied forward) - this is your "clear face" filter.
  2. Keeps only identities that end up with at least one valid cropped
     Real face AND at least one valid cropped Fake face (same "needs
     both sides" rule as build_final_dataset.py) - identities missing
     either side are excluded and reported, not silently dropped.
  3. For each remaining identity, picks exactly ONE best Real image and
     ONE best Fake image (highest face-detection confidence, tied broken
     by the largest detected face) and copies just those two into the
     final output folder.

This needs an extra one-time install that is NOT part of the main
annotation app (kept separate on purpose, so the GUI stays lightweight):

    pip install facenet-pytorch torch torchvision pillow

HOW TO USE
----------
    python crop_and_pair_faces.py

    (by default: reads FinalDataset/, writes CroppedFaces/ and
    FinalPairs/, plus a report - all next to this script)

Useful options:
    python crop_and_pair_faces.py --min-confidence 0.90 --min-face-size 60
    python crop_and_pair_faces.py --input-dir FinalDataset --output-dir FinalPairs
    python crop_and_pair_faces.py --device cuda      (if you have a GPU)
"""

import os
import sys
import shutil
import argparse
import csv

import config

VALID_EXTENSIONS = config.VALID_IMAGE_EXTENSIONS


def load_detector(device):
    try:
        from facenet_pytorch import MTCNN
    except ImportError:
        print("ERROR: facenet-pytorch is not installed.")
        print("Run this once first (in the same Python environment you use for this project):")
        print("    pip install facenet-pytorch torch torchvision")
        sys.exit(1)
    return MTCNN(keep_all=True, device=device)


def detect_best_face(detector, image_path):
    """Returns (box, confidence, pil_image). box/confidence are None if no
    face was found or the image couldn't be read. box is (x1, y1, x2, y2)."""
    from PIL import Image, UnidentifiedImageError
    try:
        image = Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None, None, None

    boxes, probs = detector.detect(image)
    if boxes is None or len(boxes) == 0:
        return None, None, image

    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    return boxes[best_idx], float(probs[best_idx]), image


def crop_with_margin(image, box, margin_ratio=0.2):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    mx, my = w * margin_ratio, h * margin_ratio
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(image.width, x2 + mx)
    y2 = min(image.height, y2 + my)
    return image.crop((int(x1), int(y1), int(x2), int(y2)))


def iter_identity_folders(input_dir):
    for entry in sorted(os.listdir(input_dir)):
        full = os.path.join(input_dir, entry)
        if os.path.isdir(full):
            yield entry, full


def run(input_dir, cropped_dir, output_dir, report_path,
        min_confidence, min_face_size, margin, detect_fn):
    """Core pipeline, separated from main() so it can be tested with a
    stand-in detect_fn (no real MTCNN needed) as well as run for real."""
    os.makedirs(cropped_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    report_rows = []
    # best[(identity_folder, type)] = (confidence, area, cropped_path, filename)
    best_per_identity_type = {}

    identity_folders = list(iter_identity_folders(input_dir))

    for idx, (identity_folder, identity_path) in enumerate(identity_folders, 1):
        for image_type in ("Real", "Fake"):
            type_dir = os.path.join(identity_path, image_type)
            if not os.path.isdir(type_dir):
                continue
            for filename in sorted(os.listdir(type_dir)):
                if not filename.lower().endswith(VALID_EXTENSIONS):
                    continue
                image_path = os.path.join(type_dir, filename)
                box, confidence, image = detect_fn(image_path)

                status = "KEPT"
                if image is None:
                    status = "DISCARDED_UNREADABLE_IMAGE"
                elif box is None:
                    status = "DISCARDED_NO_FACE_DETECTED"
                elif confidence < min_confidence:
                    status = "DISCARDED_LOW_CONFIDENCE"
                else:
                    w = box[2] - box[0]
                    h = box[3] - box[1]
                    if min(w, h) < min_face_size:
                        status = "DISCARDED_FACE_TOO_SMALL"

                cropped_path = ""
                area = 0
                if status == "KEPT":
                    cropped = crop_with_margin(image, box, margin)
                    cropped = cropped.resize((224, 224))
                    out_dir = os.path.join(cropped_dir, identity_folder, image_type)
                    os.makedirs(out_dir, exist_ok=True)
                    cropped_path = os.path.join(out_dir, filename)
                    cropped.save(cropped_path)
                    area = (box[2] - box[0]) * (box[3] - box[1])

                    key = (identity_folder, image_type)
                    current_best = best_per_identity_type.get(key)
                    if (current_best is None or confidence > current_best[0]
                            or (confidence == current_best[0] and area > current_best[1])):
                        best_per_identity_type[key] = (confidence, area, cropped_path, filename)

                report_rows.append({
                    "IdentityFolder": identity_folder, "Type": image_type, "Image": filename,
                    "Confidence": f"{confidence:.4f}" if confidence is not None else "",
                    "Status": status,
                })

    # --- Select one Real + one Fake per identity, requiring both ---
    identities_seen = sorted(set(k[0] for k in best_per_identity_type.keys()))
    excluded = []
    paired_identities = []

    for identity_folder in identities_seen:
        real_best = best_per_identity_type.get((identity_folder, "Real"))
        fake_best = best_per_identity_type.get((identity_folder, "Fake"))
        if real_best is None or fake_best is None:
            missing = [t for t, b in (("Real", real_best), ("Fake", fake_best)) if b is None]
            excluded.append((identity_folder, missing))
            continue

        for image_type, best in (("Real", real_best), ("Fake", fake_best)):
            _, _, cropped_path, filename = best
            dest_dir = os.path.join(output_dir, identity_folder, image_type)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(cropped_path, os.path.join(dest_dir, filename))
        paired_identities.append(identity_folder)

    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["IdentityFolder", "Type", "Image", "Confidence", "Status"])
        writer.writeheader()
        for row in report_rows:
            writer.writerow(row)

    return {
        "total_identities": len(identity_folders),
        "paired_identities": paired_identities,
        "excluded": excluded,
        "report_rows": report_rows,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Crop faces with MTCNN and select one Real + one Fake image per identity."
    )
    parser.add_argument("--input-dir", default=os.path.join(config.BASE_DIR, "FinalDataset"))
    parser.add_argument("--cropped-dir", default=os.path.join(config.BASE_DIR, "CroppedFaces"))
    parser.add_argument("--output-dir", default=os.path.join(config.BASE_DIR, "FinalPairs"))
    parser.add_argument("--report", default=os.path.join(config.BASE_DIR, "face_crop_report.csv"))
    parser.add_argument("--min-confidence", type=float, default=0.95,
                         help="Minimum MTCNN detection confidence to count as a 'clear face' (0-1)")
    parser.add_argument("--min-face-size", type=int, default=40,
                         help="Minimum detected face width/height in pixels")
    parser.add_argument("--margin", type=float, default=0.2,
                         help="Extra margin kept around the cropped face, as a fraction of face size")
    parser.add_argument("--device", default="cpu", help="'cpu' or 'cuda' if you have a GPU")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: input folder not found: {args.input_dir}")
        print("Run build_final_dataset.py first to produce FinalDataset/.")
        sys.exit(1)

    detector = load_detector(args.device)
    detect_fn = lambda path: detect_best_face(detector, path)

    print(f"Scanning {args.input_dir} ...")
    result = run(
        args.input_dir, args.cropped_dir, args.output_dir, args.report,
        args.min_confidence, args.min_face_size, args.margin, detect_fn,
    )

    print(f"\nDone.")
    print(f"  Identity folders scanned: {result['total_identities']}")
    print(f"  Cropped faces that passed the confidence/size check: {args.cropped_dir}")
    print(f"  Final one-Real-one-Fake pairs for {len(result['paired_identities'])} "
          f"identity(ies): {args.output_dir}")
    if result["excluded"]:
        print(f"  EXCLUDED {len(result['excluded'])} identity(ies) - missing a clear face on one side:")
        for identity_folder, missing in result["excluded"][:10]:
            print(f"    - {identity_folder}: missing {', '.join(missing)}")
        if len(result["excluded"]) > 10:
            print(f"    ... and {len(result['excluded']) - 10} more (see {args.report})")
    print(f"  Per-image detection report: {args.report}")


if __name__ == "__main__":
    main()
