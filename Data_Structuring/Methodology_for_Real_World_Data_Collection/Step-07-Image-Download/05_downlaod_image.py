"""
Step 5 - Download Images From Candidate URLs (PAIRED IDENTITIES ONLY)
======================================================================

Reads Data/Candidates/verified_faces.csv (produced by Step 4's IMAGE face check search search)
and downloads UP TO N images per (Identity_ID, Image_Type) group, where N is
a configurable target (e.g. 3 Real + 3 Fake per identity).

NEW: before downloading anything, this script builds/loads the set of
"paired" identities -- those that have at least one Real candidate AND
at least one Fake candidate. Identities that only have Real (no Fake ever
found) or only Fake are SKIPPED ENTIRELY, so you never waste download
bandwidth/time on an identity you can't actually use for a Real-vs-Fake
pair. This mirrors the Verdict column Step 3 already writes to
identity_pairing_report.csv (PAIRED / DISCARD_NO_FAKE / DISCARD_NO_REAL /
DISCARD_NEITHER_FOUND) -- if that report exists it's used directly;
otherwise it's recomputed on the fly from candidate_urls.csv.

For each identity + image type (paired identities only):
    - Try candidate URLs in Rank order (best result first).
    - Validate each download is actually a real image (content-type,
      minimum file size, minimum pixel dimensions, opens correctly).
    - Skip any candidate whose image content is a duplicate of one already
      saved for this identity/type (hash-based dedup), even if the URL
      is different.
    - Keep going until the target count is reached OR candidates run out.
    - Log how many were actually saved vs the target.

Folder layout (one folder per identity, Real/Fake nested inside):
    Data/Images/<Identity_ID>_<sanitized_name>/<Image_Type>/<seq>.<ext>

    Example:
        Data/Images/ID000001_A-Bodek/Real/01.jpg
        Data/Images/ID000001_A-Bodek/Real/02.jpg
        Data/Images/ID000001_A-Bodek/Real/03.jpg
        Data/Images/ID000001_A-Bodek/Fake/01.jpg
        Data/Images/ID000001_A-Bodek/Fake/02.jpg

Design goals (matches Step 3's philosophy):
    - Resume automatically if interrupted.
    - A (Identity_ID, Image_Type) group is terminal ("success" or
      "partial" or "no_valid_image") once ALL its candidates have been
      tried, regardless of whether the target count was fully reached --
      re-running Step 3 to fetch more candidates is what fixes "partial",
      not re-running Step 4 on the same candidate list.
    - Identities without both Real and Fake candidates are logged as
      "skipped_unpaired" ONCE and never retried on later runs (re-running
      Step 3 for that identity, then this report, is what changes that).
    - "error" groups (every candidate hit a transient network error) are
      retried on the next run.
    - Save the download log immediately after each identity/type
      (crash loses at most one identity's progress).
    - Continue even if a single download fails (logged, never crashes).
    - Rate-limited so you don't hammer image hosts.
    - Deterministic: groups processed in ascending Query_ID order,
      candidates within a group tried in ascending Rank order.

Outputs:
    Data/Images/<Identity_ID>_<name>/<Image_Type>/...  -> downloaded images
    Data/Downloads/download_log.csv                    -> one row per (Identity_ID, Image_Type) group
    Data/Logs/05_download_images.log

Config additions (auto-added to Config/config.yaml if missing):
    download_delay_seconds: 1.0
    download_timeout_seconds: 15
    min_image_bytes: 3000
    min_image_dimension: 80
    max_groups_per_run: null        # null/None = no limit
    image_targets:                  # how many images to keep per Image_Type
        Real: 10
        Fake: 10
    default_image_target: 3         # used for any Image_Type not listed above

Usage:
    python 05_download_images.py
    (run repeatedly; each run resumes from where it left off)

    python 05_download_images.py --recompute-pairs
    (ignore identity_pairing_report.csv even if present and recompute
     paired identities fresh from candidate_urls.csv -- use this if you
     ran more Step 3 batches since the report was last built)
"""

from pathlib import Path
from datetime import datetime
from io import BytesIO
import csv
import hashlib
import logging
import re
import sys
import time

import pandas as pd
import requests
import yaml
from tqdm import tqdm
from PIL import Image

# ==========================================================
# Paths
# ==========================================================
ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "Config"
DATA_DIR = ROOT / "Data"

CANDIDATES_DIR = DATA_DIR / "Candidates"
CANDIDATE_URLS_CSV = CANDIDATES_DIR / "verified_faces.csv"
IDENTITY_REPORT_CSV = CANDIDATES_DIR / "identity_pairing_report.csv"

IMAGES_DIR = DATA_DIR / "Images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOADS_DIR = DATA_DIR / "Downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_LOG_CSV = DOWNLOADS_DIR / "download_log.csv"

LOG_DIR = DATA_DIR / "Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "04_download_images.log"

# ==========================================================
# Logging (append mode -> history survives across runs)
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
CONFIG_PATH = CONFIG_DIR / "config.yaml"

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
else:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = {}

_defaults_added = False
_download_defaults = {
    "download_delay_seconds": 1.0,
    "download_timeout_seconds": 15,
    "min_image_bytes": 3000,
    "min_image_dimension": 80,
    "max_groups_per_run": None,
    "image_targets": {"Real": 3, "Fake": 3},
    "default_image_target": 3,
}
for key, value in _download_defaults.items():
    if key not in config:
        config[key] = value
        _defaults_added = True

if _defaults_added:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    logger.info(f"Added missing download settings to {CONFIG_PATH}")

DELAY_SECONDS = float(config.get("download_delay_seconds", 1.0))
TIMEOUT_SECONDS = int(config.get("download_timeout_seconds", 15))
MIN_IMAGE_BYTES = int(config.get("min_image_bytes", 3000))
MIN_IMAGE_DIMENSION = int(config.get("min_image_dimension", 80))
_max_g = config.get("max_groups_per_run", None)
MAX_GROUPS_PER_RUN = int(_max_g) if _max_g is not None else None
IMAGE_TARGETS = config.get("image_targets", {}) or {}
DEFAULT_IMAGE_TARGET = int(config.get("default_image_target", 3))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

DOWNLOAD_LOG_FIELDS = [
    "Query_ID", "Identity_ID", "Name", "Image_Type",
    "Status", "Target_Count", "Saved_Count",
    "Saved_Paths", "Used_Ranks",
    "Attempted_Candidates", "Download_Time",
]

# A group is DONE (not retried) once every candidate has been tried,
# regardless of how many images that yielded. Only "error" (every
# candidate hit a transient failure, so we never really tried them) retries.
DONE_STATUSES = {"success", "partial", "no_valid_image"}

# Identities already marked unpaired don't need to be re-logged every run.
SKIPPED_UNPAIRED_STATUS = "skipped_unpaired"


def target_for(image_type: str) -> int:
    return int(IMAGE_TARGETS.get(image_type, DEFAULT_IMAGE_TARGET))


# ==========================================================
# Pairing logic -- the whole point of this version of the script
# ==========================================================
def compute_paired_identities_from_candidates(df: pd.DataFrame) -> set:
    """Fallback: derive PAIRED identity IDs directly from candidate_urls.csv
    when identity_pairing_report.csv doesn't exist or --recompute-pairs
    was passed. An identity is PAIRED if it has >=1 Real row AND >=1 Fake
    row (any non-"real" Image_Type value counts as the fake side, matching
    Step 3's own Verdict logic)."""
    if df.empty:
        return set()

    tmp = df.copy()
    tmp["_is_real"] = tmp["Image_Type"].str.strip().str.lower().eq("real")
    tmp["_is_fake"] = ~tmp["_is_real"]

    grouped = tmp.groupby("Identity_ID").agg(
        Real_Count=("_is_real", "sum"),
        Fake_Count=("_is_fake", "sum"),
    )
    paired = grouped[(grouped["Real_Count"] > 0) & (grouped["Fake_Count"] > 0)]
    return set(paired.index.astype(str))


def load_paired_identities(df: pd.DataFrame, force_recompute: bool) -> set:
    """Prefer the report Step 3 already builds; fall back to recomputing."""
    if not force_recompute and IDENTITY_REPORT_CSV.exists():
        try:
            report = pd.read_csv(IDENTITY_REPORT_CSV, dtype=str)
            if not report.empty and "Verdict" in report.columns:
                paired = report[report["Verdict"] == "PAIRED"]
                ids = set(paired["Identity_ID"].astype(str))
                logger.info(
                    f"Loaded {len(ids)} PAIRED identities from "
                    f"{IDENTITY_REPORT_CSV.name}"
                )
                return ids
        except Exception as e:
            logger.warning(
                f"Could not read {IDENTITY_REPORT_CSV.name} ({e}) -- "
                "recomputing pairing from candidate_urls.csv instead."
            )

    ids = compute_paired_identities_from_candidates(df)
    logger.info(f"Computed {len(ids)} PAIRED identities directly from candidate_urls.csv")
    return ids


# ==========================================================
# Helpers
# ==========================================================
def sanitize_filename(name: str) -> str:
    """Turn a person's name into a safe filename fragment."""
    name = str(name).strip()
    name = re.sub(r"[^\w\s-]", "", name)   # drop punctuation
    name = re.sub(r"\s+", "-", name)       # spaces -> dashes
    return name or "unknown"


def load_completed_groups() -> set:
    """Return set of (Identity_ID, Image_Type) already terminally resolved.

    'error' rows are excluded so they get retried on the next run.
    """
    if not DOWNLOAD_LOG_CSV.exists():
        return set()
    try:
        df = pd.read_csv(DOWNLOAD_LOG_CSV, dtype=str)
    except pd.errors.EmptyDataError:
        return set()
    if df.empty or "Status" not in df.columns:
        return set()
    done_mask = df["Status"].isin(DONE_STATUSES | {SKIPPED_UNPAIRED_STATUS})
    done = df.loc[done_mask, ["Identity_ID", "Image_Type"]]
    return set(zip(done["Identity_ID"].astype(str), done["Image_Type"].astype(str)))


def load_already_flagged_unpaired_identities() -> set:
    """Identity_IDs already logged as skipped_unpaired -- don't re-log every run."""
    if not DOWNLOAD_LOG_CSV.exists():
        return set()
    try:
        df = pd.read_csv(DOWNLOAD_LOG_CSV, dtype=str)
    except pd.errors.EmptyDataError:
        return set()
    if df.empty or "Status" not in df.columns:
        return set()
    flagged = df[df["Status"] == SKIPPED_UNPAIRED_STATUS]
    return set(flagged["Identity_ID"].astype(str))


def append_download_log(record: dict) -> None:
    file_exists = DOWNLOAD_LOG_CSV.exists()
    with open(DOWNLOAD_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DOWNLOAD_LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def guess_extension(url: str, content_type: str) -> str:
    content_type = (content_type or "").lower()
    if "jpeg" in content_type or "jpg" in content_type:
        return ".jpg"
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    if "gif" in content_type:
        return ".gif"
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        if url.lower().split("?")[0].endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"  # safe default


def existing_hashes(folder: Path) -> set:
    """Hash any images already saved in this folder (from a prior partial run)."""
    hashes = set()
    if not folder.exists():
        return hashes
    for p in folder.iterdir():
        if p.is_file():
            try:
                hashes.add(hashlib.md5(p.read_bytes()).hexdigest())
            except Exception:
                pass
    return hashes


def next_seq_number(folder: Path) -> int:
    """Find the next free sequence number so resumed runs don't overwrite files."""
    if not folder.exists():
        return 1
    existing = [p.stem for p in folder.iterdir() if p.is_file()]
    nums = [int(s) for s in existing if s.isdigit()]
    return (max(nums) + 1) if nums else 1


def try_download_one(url: str) -> tuple:
    """Attempt to download and validate a single candidate image URL.

    Returns (success, content_bytes_or_None, extension_or_None, reason).
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS, stream=True)
    except Exception as e:
        return False, None, None, f"request_error: {e}"

    if resp.status_code != 200:
        return False, None, None, f"http_{resp.status_code}"

    content_type = resp.headers.get("Content-Type", "")

    # Reading the body is where connections most often die mid-stream
    # (ConnectionResetError, ChunkedEncodingError, IncompleteRead, etc).
    # This must be caught too, or one flaky host crashes the entire run.
    try:
        content = resp.content
    except Exception as e:
        return False, None, None, f"request_error: {e}"
    finally:
        try:
            resp.close()
        except Exception:
            pass

    if len(content) < MIN_IMAGE_BYTES:
        return False, None, None, "too_small_bytes"

    try:
        img = Image.open(BytesIO(content))
        img.verify()
        img2 = Image.open(BytesIO(content))
        width, height = img2.size
    except Exception as e:
        return False, None, None, f"not_a_valid_image: {e}"

    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        return False, None, None, "too_small_dimensions"

    ext = guess_extension(url, content_type)
    return True, content, ext, "ok"


# ==========================================================
# Main
# ==========================================================
def main(force_recompute_pairs: bool = False) -> None:
    if not CANDIDATE_URLS_CSV.exists():
        logger.error(f"Candidate URLs file not found: {CANDIDATE_URLS_CSV}")
        print(f"ERROR: Candidate URLs file not found: {CANDIDATE_URLS_CSV}")
        print("Run Step 3 first (make sure it's the IMAGE search version).")
        return

    df = pd.read_csv(CANDIDATE_URLS_CSV, dtype=str)
    if "Image_URL" not in df.columns:
        logger.error("candidate_urls.csv has no 'Image_URL' column -- "
                      "this looks like it came from a web search, not an image search.")
        print("ERROR: candidate_urls.csv has no 'Image_URL' column.")
        print("This file looks like it came from a regular web search, not "
              "Serper's image search. Re-run Step 4 (image search version) first.")
        return

    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df = df.dropna(subset=["Image_URL"])
    df = df[df["Image_URL"].str.strip() != ""]

    # --- THE KEY FILTER: only identities with both Real and Fake candidates ---
    paired_ids = load_paired_identities(df, force_recompute=force_recompute_pairs)
    all_ids = set(df["Identity_ID"].astype(str).unique())
    unpaired_ids = all_ids - paired_ids

    print(f"Identities total          : {len(all_ids)}")
    print(f"Identities PAIRED (both)  : {len(paired_ids)}  <- these get downloaded")
    print(f"Identities unpaired       : {len(unpaired_ids)}  <- these are SKIPPED")

    # Log each newly-seen unpaired identity once so future runs don't
    # even bother re-checking it (until you re-run Step 3 for it and the
    # pairing report/candidate file changes -- at which point delete its
    # skipped_unpaired row, or just let --recompute-pairs re-evaluate it).
    already_flagged = load_already_flagged_unpaired_identities()
    newly_unpaired = unpaired_ids - already_flagged
    for identity_id in sorted(newly_unpaired):
        name_series = df.loc[df["Identity_ID"] == identity_id, "Name"]
        name = name_series.iloc[0] if not name_series.empty else ""
        append_download_log({
            "Query_ID": "",
            "Identity_ID": identity_id,
            "Name": name,
            "Image_Type": "",
            "Status": SKIPPED_UNPAIRED_STATUS,
            "Target_Count": "",
            "Saved_Count": 0,
            "Saved_Paths": "",
            "Used_Ranks": "",
            "Attempted_Candidates": 0,
            "Download_Time": datetime.now().isoformat(timespec="seconds"),
        })
        logger.info(f"{identity_id} | Skipped -- no Real/Fake pair (Real-only or Fake-only)")

    # Restrict everything downstream to paired identities only.
    df = df[df["Identity_ID"].astype(str).isin(paired_ids)].reset_index(drop=True)

    if df.empty:
        print("=" * 60)
        print("No paired identities to download -- nothing to do.")
        print("Every identity currently has only Real or only Fake candidates.")
        print("Run Step 3 again (more batches / broader queries) to find the missing side.")
        print("=" * 60)
        return

    group_cols = ["Query_ID", "Identity_ID", "Name", "Image_Type"]
    groups = df.groupby(group_cols, sort=True)

    completed = load_completed_groups()

    pending_groups = []
    for keys, group_df in groups:
        query_id, identity_id, name, image_type = keys
        if (str(identity_id), str(image_type)) in completed:
            continue
        pending_groups.append((keys, group_df.sort_values("Rank")))

    pending_groups.sort(key=lambda x: str(x[0][0]))  # sort by Query_ID

    total_groups = df.groupby(group_cols).ngroups
    if MAX_GROUPS_PER_RUN is not None:
        batch = pending_groups[:MAX_GROUPS_PER_RUN]
    else:
        batch = pending_groups

    logger.info("=" * 60)
    logger.info("STEP 4 START (Image Download, paired identities only)")
    logger.info(f"Paired identities          : {len(paired_ids)}")
    logger.info(f"Unpaired identities skipped: {len(unpaired_ids)}")
    logger.info(f"Total identity/type groups : {total_groups}")
    logger.info(f"Already completed          : {len(completed)}")
    logger.info(f"Remaining pending          : {len(pending_groups)}")
    logger.info(f"Groups this run            : {len(batch)}")
    logger.info("=" * 60)

    if not batch:
        print("=" * 60)
        print("STEP 4 - Nothing to do")
        print("=" * 60)
        print(f"Total identity/type groups : {total_groups}")
        print(f"Already completed          : {len(completed)}")
        print("Every paired identity/type has already been processed.")
        print("=" * 60)
        return

    n_success = 0
    n_partial = 0
    n_discarded = 0
    n_error = 0

    for keys, group_df in tqdm(batch, total=len(batch)):
        query_id, identity_id, name, image_type = keys
        query_id = str(query_id)
        identity_id = str(identity_id)
        image_type = str(image_type)

        target = target_for(image_type)

        identity_folder = f"{identity_id}_{sanitize_filename(name)}"
        type_dir = IMAGES_DIR / identity_folder / image_type
        type_dir.mkdir(parents=True, exist_ok=True)

        seen_hashes = existing_hashes(type_dir)
        seq = next_seq_number(type_dir)
        saved_count = len(seen_hashes)

        saved_paths = []
        used_ranks = []
        attempted = 0
        had_transient_error = False

        for _, row in group_df.iterrows():
            if saved_count >= target:
                break  # already have enough for this identity/type

            attempted += 1
            url = str(row["Image_URL"]).strip()
            if not url:
                continue

            ok, content, ext, reason = try_download_one(url)
            time.sleep(DELAY_SECONDS)

            if not ok:
                logger.info(
                    f"{query_id} | {identity_id} | {image_type} | "
                    f"rank {row.get('Rank', '?')} failed ({reason}), trying next candidate"
                )
                if reason.startswith("request_error"):
                    had_transient_error = True
                continue

            content_hash = hashlib.md5(content).hexdigest()
            if content_hash in seen_hashes:
                logger.info(
                    f"{query_id} | {identity_id} | {image_type} | "
                    f"rank {row.get('Rank', '?')} duplicate image, skipping"
                )
                continue

            out_path = type_dir / f"{seq:02d}{ext}"
            try:
                with open(out_path, "wb") as f:
                    f.write(content)
                seen_hashes.add(content_hash)
                saved_paths.append(str(out_path))
                used_ranks.append(str(row.get("Rank", "")))
                saved_count += 1
                seq += 1
                logger.info(
                    f"{query_id} | {identity_id} | {image_type} | "
                    f"Saved rank {row.get('Rank', '?')} -> {out_path} "
                    f"({saved_count}/{target})"
                )
            except Exception as e:
                logger.warning(f"{query_id} | Failed to write file for {url}: {e}")

        if saved_count >= target:
            status = "success"
            n_success += 1
        elif saved_count > 0:
            status = "partial"
            n_partial += 1
            logger.info(
                f"{query_id} | {identity_id} | {image_type} | "
                f"Partial -- only {saved_count}/{target} found among "
                f"{len(group_df)} candidates"
            )
        else:
            if had_transient_error and attempted == len(group_df):
                status = "error"
                n_error += 1
                logger.warning(
                    f"{query_id} | {identity_id} | {image_type} | "
                    f"All {attempted} candidates hit transient errors -- will retry next run"
                )
            else:
                status = "no_valid_image"
                n_discarded += 1
                logger.info(
                    f"{query_id} | {identity_id} | {image_type} | "
                    f"Discarded -- no valid image found among {attempted} candidates"
                )

        append_download_log({
            "Query_ID": query_id,
            "Identity_ID": identity_id,
            "Name": name,
            "Image_Type": image_type,
            "Status": status,
            "Target_Count": target,
            "Saved_Count": saved_count,
            "Saved_Paths": "|".join(saved_paths),
            "Used_Ranks": "|".join(used_ranks),
            "Attempted_Candidates": attempted,
            "Download_Time": datetime.now().isoformat(timespec="seconds"),
        })

    remaining_after = len(pending_groups) - (n_success + n_partial + n_discarded)

    logger.info("=" * 60)
    logger.info("STEP 4 RUN COMPLETE")
    logger.info(f"Success (target reached) : {n_success}")
    logger.info(f"Partial (some, not all)  : {n_partial}")
    logger.info(f"Discarded (no image)     : {n_discarded}")
    logger.info(f"Errors (will retry)      : {n_error}")
    logger.info(f"Remaining                : {remaining_after}")
    logger.info("=" * 60)

    print("=" * 60)
    print("STEP 4 RUN COMPLETE")
    print("=" * 60)
    print(f"Paired identities (of {len(all_ids)}) : {len(paired_ids)}")
    print(f"Total identity/type groups : {total_groups}")
    print(f"Processed this run         : {len(batch)}")
    print(f"  Success (target reached) : {n_success}")
    print(f"  Partial (some, not all)  : {n_partial}")
    print(f"  Discarded (no image)     : {n_discarded}")
    print(f"  Errors (will retry)      : {n_error}")
    print(f"Remaining pending groups   : {remaining_after}")
    print(f"Images saved under         : {IMAGES_DIR}")
    print(f"Download log               : {DOWNLOAD_LOG_CSV}")
    print(f"Log file                   : {LOG_FILE}")
    print("=" * 60)
    if remaining_after > 0:
        print("Run the script again to continue with the next batch.")
    if n_partial > 0:
        print(f"Note: {n_partial} group(s) got fewer images than the target -- "
              "consider re-running Step 3 for those to gather more candidates.")


if __name__ == "__main__":
    recompute = "--recompute-pairs" in sys.argv
    try:
        main(force_recompute_pairs=recompute)
    except KeyboardInterrupt:
        logger.warning("Run interrupted by user (KeyboardInterrupt). "
                        "Progress up to the last completed identity/type has been saved. "
                        "Re-run the script to resume.")
        print("\nInterrupted. Progress saved - re-run the script to resume.")