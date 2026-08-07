"""
Step 4 - Face Verification of Candidate Images (MTCNN)
=======================================================

This is the ONLY stage that actually opens/downloads the image bytes.
Step 3 filters by URL/domain/title metadata; this stage checks the pixels.

For every row in Data/Candidates/candidate_urls.csv, this script:
    1. Downloads the image (skips/retries gracefully on failure).
    2. Runs MTCNN face detection (facenet-pytorch).
    3. Rejects the image if:
         - no face is found                      -> "no_face"
         - more than one face is found            -> "multiple_faces"
         - the largest face's bounding box is too -> "face_too_small"
           small relative to the image (below
           min_face_area_ratio) OR below an
           absolute pixel floor (min_face_pixels)
    4. Accepts (status "ok") only single, sufficiently large faces.

Resumable / append-only, same pattern as Step 3:
    - A ledger (face_check_log.csv) tracks every Image_URL already
      processed with a terminal status so re-runs skip finished work.
    - "download_error" and "detector_error" are retried on the next run.
    - Accepted rows are appended to verified_faces.csv immediately.

OUTPUTS
-------
    Data/Candidates/verified_faces.csv     -> accepted candidates only, with face metadata
    Data/Candidates/face_check_log.csv     -> per-Image_URL status ledger (append-only)
    Data/Logs/04_face_verify_candidates.log

CONFIG additions (Config/config.yaml)
--------------------------------------
    min_face_area_ratio: 0.03     # face bbox area / image area must be >= this
    min_face_pixels: 60           # AND face bbox width/height must both be >= this (px)
    download_timeout_seconds: 15
    face_verify_delay_seconds: 0.3
    max_face_checks_per_run: 500

REQUIREMENTS
------------
    pip install facenet-pytorch torch torchvision pillow pandas pyyaml tqdm requests

USAGE
-----
    python 04_face_verify_candidates.py
        -> processes the next batch of unverified candidate_urls.csv rows

    python 04_face_verify_candidates.py --identity-type fake
        -> only process rows where Image_Type != "real" (skip real photos,
           useful since real photos are usually already known-good)
"""

from pathlib import Path
from datetime import datetime
from io import BytesIO
import csv
import logging
import sys
import time

import pandas as pd
import requests
import yaml
from PIL import Image
from tqdm import tqdm

try:
    import torch
    from facenet_pytorch import MTCNN
except ImportError:
    print("ERROR: facenet-pytorch / torch not installed.")
    print("Run: pip install facenet-pytorch torch torchvision")
    sys.exit(1)

# ==========================================================
# Paths
# ==========================================================
ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "Config"
DATA_DIR = ROOT / "Data"
CANDIDATES_DIR = DATA_DIR / "Candidates"

CANDIDATE_URLS_CSV = CANDIDATES_DIR / "candidate_urls.csv"
VERIFIED_FACES_CSV = CANDIDATES_DIR / "verified_faces.csv"
FACE_CHECK_LOG_CSV = CANDIDATES_DIR / "face_check_log.csv"

LOG_DIR = DATA_DIR / "Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "04_face_verify_candidates.log"

CONFIG_PATH = CONFIG_DIR / "config.yaml"

# ==========================================================
# Logging
# ==========================================================
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    filemode="a",
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("step4")

_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
logger.addHandler(_console)

# ==========================================================
# Config
# ==========================================================
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
else:
    config = {}

MIN_FACE_AREA_RATIO = float(config.get("min_face_area_ratio", 0.03))
MIN_FACE_PIXELS = int(config.get("min_face_pixels", 60))
DOWNLOAD_TIMEOUT = int(config.get("download_timeout_seconds", 15))
DELAY_SECONDS = float(config.get("face_verify_delay_seconds", 0.3))
_max_batch = config.get("max_face_checks_per_run", 500)
MAX_PER_RUN = int(_max_batch) if _max_batch is not None else None

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DatasetBot/1.0; +face-verification-stage)"
}

VERIFIED_FIELDS = [
    "Query_ID", "Identity_ID", "Name", "Image_Type", "Search_Query",
    "Rank", "Title", "Image_URL", "Source_Page_URL", "Display_Link",
    "Trust_Tier", "Score", "Face_Count", "Face_Box", "Face_Area_Ratio",
]

LOG_FIELDS = ["Image_URL", "Status", "Checked_Time", "Face_Count", "Detail"]

TERMINAL_STATUSES = {"ok", "no_face", "multiple_faces", "face_too_small", "invalid_image"}
# "download_error" and "detector_error" are NOT terminal -- retried next run.

# ==========================================================
# MTCNN setup (single shared detector instance)
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DETECTOR = MTCNN(keep_all=True, device=DEVICE)
logger.info(f"MTCNN initialized on device: {DEVICE}")


# ==========================================================
# Helpers
# ==========================================================
def load_checked_urls() -> set:
    if not FACE_CHECK_LOG_CSV.exists():
        return set()
    try:
        df = pd.read_csv(FACE_CHECK_LOG_CSV, dtype=str)
    except pd.errors.EmptyDataError:
        return set()
    if df.empty or "Status" not in df.columns:
        return set()
    done = df[df["Status"].isin(TERMINAL_STATUSES)]
    return set(done["Image_URL"].astype(str))


def append_log(record: dict) -> None:
    file_exists = FACE_CHECK_LOG_CSV.exists()
    with open(FACE_CHECK_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def append_verified(row: dict) -> None:
    file_exists = VERIFIED_FACES_CSV.exists()
    with open(VERIFIED_FACES_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VERIFIED_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def download_image(url: str) -> Image.Image:
    resp = requests.get(url, headers=REQUEST_HEADERS, timeout=DOWNLOAD_TIMEOUT)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content))
    img = img.convert("RGB")
    return img


def evaluate_faces(img: Image.Image):
    """
    Returns (status, face_count, best_box, area_ratio, detail)
    best_box is (x1, y1, x2, y2) of the largest detected face, or None.
    """
    width, height = img.size
    image_area = float(width * height)

    boxes, probs = DETECTOR.detect(img)

    if boxes is None or len(boxes) == 0:
        return "no_face", 0, None, 0.0, "MTCNN found no faces"

    if len(boxes) > 1:
        return "multiple_faces", len(boxes), None, 0.0, f"MTCNN found {len(boxes)} faces"

    x1, y1, x2, y2 = boxes[0]
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    box_area = box_w * box_h
    area_ratio = box_area / image_area if image_area > 0 else 0.0

    if box_w < MIN_FACE_PIXELS or box_h < MIN_FACE_PIXELS or area_ratio < MIN_FACE_AREA_RATIO:
        detail = (f"face {box_w:.0f}x{box_h:.0f}px, area_ratio={area_ratio:.4f} "
                  f"(min_px={MIN_FACE_PIXELS}, min_ratio={MIN_FACE_AREA_RATIO})")
        return "face_too_small", 1, (x1, y1, x2, y2), area_ratio, detail

    return "ok", 1, (x1, y1, x2, y2), area_ratio, "single face, passes size checks"


# ==========================================================
# Main
# ==========================================================
def main(identity_type_filter: str = None) -> None:
    if not CANDIDATE_URLS_CSV.exists():
        print(f"ERROR: {CANDIDATE_URLS_CSV} not found. Run Step 3 first.")
        return

    df = pd.read_csv(CANDIDATE_URLS_CSV, dtype=str)
    if df.empty:
        print("candidate_urls.csv is empty -- nothing to verify.")
        return

    if identity_type_filter:
        df = df[df["Image_Type"].str.strip().str.lower() == identity_type_filter.strip().lower()]

    df = df.drop_duplicates(subset=["Image_URL"]).reset_index(drop=True)

    checked = load_checked_urls()
    pending = df[~df["Image_URL"].isin(checked)].reset_index(drop=True)
    batch = pending.head(MAX_PER_RUN) if MAX_PER_RUN is not None else pending

    logger.info("=" * 60)
    logger.info("STEP 4 START (Face Verification)")
    logger.info(f"Total candidates     : {len(df)}")
    logger.info(f"Already checked      : {len(checked)}")
    logger.info(f"Remaining pending    : {len(pending)}")
    logger.info(f"This run             : {len(batch)}")
    logger.info(f"Device               : {DEVICE}")
    logger.info("=" * 60)

    if batch.empty:
        print("Nothing to do -- all candidates already face-checked.")
        return

    n_ok = n_no_face = n_multi = n_small = n_dl_err = n_det_err = n_invalid = 0

    for _, row in tqdm(batch.iterrows(), total=len(batch)):
        image_url = row["Image_URL"]
        checked_time = datetime.now().isoformat(timespec="seconds")

        try:
            img = download_image(image_url)
        except Exception as e:
            logger.warning(f"Download failed: {image_url} | {e}")
            append_log({
                "Image_URL": image_url, "Status": "download_error",
                "Checked_Time": checked_time, "Face_Count": 0, "Detail": str(e)[:200],
            })
            n_dl_err += 1
            time.sleep(DELAY_SECONDS)
            continue

        try:
            status, face_count, box, area_ratio, detail = evaluate_faces(img)
        except Exception as e:
            logger.exception(f"Detector failed: {image_url} | {e}")
            append_log({
                "Image_URL": image_url, "Status": "detector_error",
                "Checked_Time": checked_time, "Face_Count": 0, "Detail": str(e)[:200],
            })
            n_det_err += 1
            time.sleep(DELAY_SECONDS)
            continue

        append_log({
            "Image_URL": image_url, "Status": status,
            "Checked_Time": checked_time, "Face_Count": face_count, "Detail": detail,
        })

        if status == "ok":
            append_verified({
                "Query_ID": row.get("Query_ID", ""),
                "Identity_ID": row.get("Identity_ID", ""),
                "Name": row.get("Name", ""),
                "Image_Type": row.get("Image_Type", ""),
                "Search_Query": row.get("Search_Query", ""),
                "Rank": row.get("Rank", ""),
                "Title": row.get("Title", ""),
                "Image_URL": image_url,
                "Source_Page_URL": row.get("Source_Page_URL", ""),
                "Display_Link": row.get("Display_Link", ""),
                "Trust_Tier": row.get("Trust_Tier", ""),
                "Score": row.get("Score", ""),
                "Face_Count": face_count,
                "Face_Box": str(box),
                "Face_Area_Ratio": round(area_ratio, 5),
            })
            n_ok += 1
        elif status == "no_face":
            n_no_face += 1
        elif status == "multiple_faces":
            n_multi += 1
        elif status == "face_too_small":
            n_small += 1
        elif status == "invalid_image":
            n_invalid += 1

        time.sleep(DELAY_SECONDS)

    logger.info("=" * 60)
    logger.info("STEP 4 RUN COMPLETE")
    logger.info(f"OK (accepted)     : {n_ok}")
    logger.info(f"No face           : {n_no_face}")
    logger.info(f"Multiple faces    : {n_multi}")
    logger.info(f"Face too small    : {n_small}")
    logger.info(f"Download errors   : {n_dl_err} (will retry)")
    logger.info(f"Detector errors   : {n_det_err} (will retry)")
    logger.info("=" * 60)

    print("=" * 60)
    print("STEP 4 RUN COMPLETE")
    print("=" * 60)
    print(f"Accepted (ok)        : {n_ok}")
    print(f"Rejected - no face   : {n_no_face}")
    print(f"Rejected - multi face: {n_multi}")
    print(f"Rejected - too small : {n_small}")
    print(f"Download errors      : {n_dl_err}  <- will retry next run")
    print(f"Detector errors      : {n_det_err}  <- will retry next run")
    print(f"Verified faces file  : {VERIFIED_FACES_CSV}")
    print(f"Face check log       : {FACE_CHECK_LOG_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    filt = None
    if "--identity-type" in sys.argv:
        idx = sys.argv.index("--identity-type")
        if idx + 1 < len(sys.argv):
            filt = sys.argv[idx + 1]
    try:
        main(identity_type_filter=filt)
    except KeyboardInterrupt:
        logger.warning("Run interrupted by user. Progress saved -- re-run to resume.")
        print("\nInterrupted. Progress saved - re-run the script to resume.")