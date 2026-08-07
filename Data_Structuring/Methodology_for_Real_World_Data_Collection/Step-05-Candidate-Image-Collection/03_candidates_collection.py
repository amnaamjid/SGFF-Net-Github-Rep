"""
Step 3 - Collect Candidate IMAGE URLs (Serper.dev Image Search)
============================================================

WHAT THIS STAGE CAN AND CANNOT VERIFY
--------------------------------------
This script filters candidates using URL-level and metadata-level signals
only (domain trust tier, file extension, title/URL keyword scoring, which
query/Image_Type produced the result). It does NOT open or analyze the
images themselves -- that face-verification pass lives in
04_face_verify_candidates.py, which downloads each accepted candidate and
runs MTCNN face detection.

What this script does for REAL queries:
    - Restricted to trusted-domain results (news wires, Getty, Wikimedia
      Commons, official/gov/edu sites, major outlets), so you are not
      downloading Pinterest re-posts or fan edits as "real".

What this script does for FAKE queries:
    - Relies on the AI/deepfake-specific query wording already in
      search_queries.csv (e.g. "<Name> AI generated", "<Name> deepfake",
      "<Name> synthetic portrait").
    - Explicitly EXCLUDES trusted news/editorial/stock-photo domains, so a
      genuine Reuters or Shutterstock photo can never get counted as a
      "fake" candidate by accident.
    - Scores every remaining candidate on title keywords, URL keywords,
      source-page-URL keywords, and a domain whitelist of sites that
      routinely host AI-generated images (Lexica, Civitai, etc). Only
      candidates clearing FAKE_SCORE_THRESHOLD are accepted, and when a
      query returns more acceptable hits than open slots, the
      highest-scoring ones are kept first.

Only direct, downloadable image files (.jpg/.jpeg/.png) are kept.

This is a pre-filter, not a guarantee. The face-verification pass
(04_face_verify_candidates.py) is the next real check ("is this actually a
usable single-face photo"); manual spot-checking for "is this actually
synthetic" remains the final human step.

DESIGN GOALS (final version - do not redesign further)
--------------------------------------------------------
    - Resume automatically if interrupted.
    - Skip queries already completed ("success" / "no_results" / "skipped_target_met").
    - Retry queries that previously errored ("error" is NOT skipped).
    - Save results immediately after each query (crash loses at most 1 query).
    - Per-identity, per-type (Real/Fake) TARGET COUNT: once an identity has
      enough accepted candidates of a given type, remaining queries of that
      type for that identity are skipped WITHOUT spending an API call.
    - Wait `serper_delay_seconds` between requests.
    - Continue even if a single query fails.
    - Persistent, append-only log across runs.
    - Never overwrite previous results (all CSVs are append-only).
    - Deterministic: queries processed in ascending Query_ID order.
    - After each run, (re)build an identity-level pairing report so you can
      see, at a glance, which identities have both Real and Fake coverage
      and which should be discarded.

OUTPUTS
-------
    Data/Candidates/candidate_urls.csv        -> one row per accepted image (now includes Score)
    Data/Candidates/searched_queries.csv      -> one row per query attempt (status ledger)
    Data/Candidates/identity_pairing_report.csv -> per-identity Real/Fake counts + verdict
    Data/Logs/03_collect_candidate_urls.log

CONFIG
------
    Config/api_keys.yaml
        serper_api_key: <your key from serper.dev, free, no credit card>

    Config/config.yaml   (auto-created with defaults if missing)
        serper_results_per_query: 50
        serper_delay_seconds: 1.5
        max_queries_per_run: 100
        real_images_target_per_identity: 10
        fake_images_target_per_identity: 10
        allow_tier2_real_sources: true
        fake_score_threshold: 2       # minimum score for a Fake candidate to be accepted
        fake_title_keyword_weight: 3
        fake_url_keyword_weight: 2
        fake_source_url_keyword_weight: 2
        fake_whitelist_domain_bonus: 5

USAGE
-----
    python 03_collect_candidate_urls.py
        -> runs the next batch of pending queries, then refreshes the report.

    python 03_collect_candidate_urls.py --report-only
        -> skips searching entirely, just rebuilds identity_pairing_report.csv
           from whatever is already in candidate_urls.csv. Safe to run anytime.
"""

from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
import csv
import logging
import sys
import time
from collections import defaultdict

import pandas as pd
import requests
import yaml
from tqdm import tqdm

# ==========================================================
# Paths
# ==========================================================
ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "Config"
DATA_DIR = ROOT / "Data"

SEARCH_QUERY_CSV = DATA_DIR / "Search" / "search_queries.csv"

CANDIDATES_DIR = DATA_DIR / "Candidates"
CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATE_URLS_CSV = CANDIDATES_DIR / "candidate_urls.csv"
SEARCHED_QUERIES_CSV = CANDIDATES_DIR / "searched_queries.csv"
IDENTITY_REPORT_CSV = CANDIDATES_DIR / "identity_pairing_report.csv"

LOG_DIR = DATA_DIR / "Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "03_collect_candidate_urls.log"

# ==========================================================
# Logging (append mode -> history survives across runs)
# ==========================================================
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    filemode="a",
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("step3")

_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
logger.addHandler(_console)

# ==========================================================
# Config
# ==========================================================
API_KEYS_PATH = CONFIG_DIR / "api_keys.yaml"
CONFIG_PATH = CONFIG_DIR / "config.yaml"

if not API_KEYS_PATH.exists():
    raise FileNotFoundError(
        f"Missing API key file: {API_KEYS_PATH}\n"
        f"Expected key: serper_api_key\n"
        f"Get a free key (no credit card) at https://serper.dev"
    )

with open(API_KEYS_PATH, "r", encoding="utf-8") as f:
    api = yaml.safe_load(f) or {}

if "serper_api_key" not in api:
    raise KeyError(
        "Config/api_keys.yaml is missing 'serper_api_key'.\n"
        "Sign up for free at https://serper.dev, copy your key, and add:\n"
        '  serper_api_key: "your-key-here"\n'
        "to Config/api_keys.yaml"
    )

API_KEY = api["serper_api_key"]

if not CONFIG_PATH.exists():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    default_config = {
        "serper_results_per_query": 10,
        "serper_delay_seconds": 1.5,
        "max_queries_per_run": 100,
        "real_images_target_per_identity": 5,
        "fake_images_target_per_identity": 5,
        "allow_tier2_real_sources": True,
        "fake_score_threshold": 3,
        "fake_title_keyword_weight": 3,
        "fake_url_keyword_weight": 2,
        "fake_source_url_keyword_weight": 2,
        "fake_whitelist_domain_bonus": 5,
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(default_config, f, sort_keys=False)
    config = default_config
    logger.info(f"Config not found. Created default config at {CONFIG_PATH}")
else:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

RESULTS_PER_QUERY = int(config.get("serper_results_per_query", 10))
DELAY_SECONDS = float(config.get("serper_delay_seconds", 1.5))
_max_q = config.get("max_queries_per_run", 100)
MAX_QUERIES_PER_RUN = int(_max_q) if _max_q is not None else None
REAL_TARGET = int(config.get("real_images_target_per_identity", 5))
FAKE_TARGET = int(config.get("fake_images_target_per_identity", 5))
ALLOW_TIER2_REAL = bool(config.get("allow_tier2_real_sources", True))

FAKE_SCORE_THRESHOLD = float(config.get("fake_score_threshold", 3))
W_TITLE = float(config.get("fake_title_keyword_weight", 3))
W_URL = float(config.get("fake_url_keyword_weight", 2))
W_SOURCE_URL = float(config.get("fake_source_url_keyword_weight", 2))
W_WHITELIST = float(config.get("fake_whitelist_domain_bonus", 5))

URL = "https://google.serper.dev/images"  # /images -> real downloadable image URLs
HEADERS = {
    "X-API-KEY": API_KEY,
    "Content-Type": "application/json",
}

CANDIDATE_FIELDS = [
    "Query_ID", "Identity_ID", "Name", "Image_Type",
    "Search_Query", "Rank", "Title", "Image_URL",
    "Source_Page_URL", "Display_Link", "Thumbnail_URL", "Trust_Tier",
    "Score",
]

SEARCHED_FIELDS = ["Query_ID", "Status", "Search_Time", "Results_Returned"]

# Statuses that mean "do not search this query again"
DONE_STATUSES = {"success", "no_results", "skipped_target_met", "blocked_query_syntax"}

ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")

# ==========================================================
# Source trust tiers (edit freely -- these are starting points)
# ==========================================================
REAL_TIER1_DOMAINS = {
    "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com", "ap.org",
    "gettyimages.com", "media.gettyimages.com", "wikimedia.org",
    "commons.wikimedia.org", "wikipedia.org", "npr.org", "nytimes.com",
    "theguardian.com", "washingtonpost.com", "forbes.com", "time.com",
    "usatoday.com", "cnn.com", "abcnews.go.com", "cbsnews.com",
    "nbcnews.com", "espn.com", "variety.com", "hollywoodreporter.com",
}

# Any domain ending in these suffixes is automatically Tier 1 / always excluded from Fake
REAL_TIER1_SUFFIXES = (".gov", ".edu", ".ac.uk")

# Domains never accepted as "Real" evidence (low provenance / user-uploaded)
REAL_BLOCKED_DOMAINS = {
    "pinterest.com", "pin.it", "imgur.com", "reddit.com", "quora.com",
}

# Stock/editorial photo houses -- these are (almost) never AI-generated,
# so they are blacklisted from Fake results in addition to the news wires.
STOCK_PHOTO_DOMAINS = {
    "shutterstock.com", "alamy.com", "istockphoto.com", "dreamstime.com",
    "123rf.com", "depositphotos.com", "stocksy.com",
}

# Never let a Fake candidate come from a genuine editorial / stock source
FAKE_EXCLUDED_DOMAINS = REAL_TIER1_DOMAINS | STOCK_PHOTO_DOMAINS

# Domains that routinely host AI-generated / synthetic imagery -- big score bonus
FAKE_WHITELIST_DOMAINS = {
    "lexica.art", "civitai.com", "playgroundai.com", "playground.com",
    "huggingface.co", "nightcafe.studio", "creator.nightcafe.studio",
    "tensor.art", "openart.ai", "leonardo.ai", "app.leonardo.ai",
    "prompthero.com",
}

# Keyword lists for scoring (all lowercase, matched as substrings)
AI_TITLE_KEYWORDS = [
    "ai generated", "ai-generated", "aigenerated", "generated by ai",
    "artificial intelligence generated", "synthetic", "deepfake", "deep fake",
    "stable diffusion", "midjourney", "dall-e", "dall e", "dalle",
    "flux ai", "ai art", "ai image", "ai portrait", "gan generated",
    "machine generated", "diffusion model",
]

AI_URL_KEYWORDS = [
    "ai-generated", "aigenerated", "ai_generated", "deepfake", "deep-fake",
    "synthetic", "stable-diffusion", "stablediffusion", "midjourney",
    "dall-e", "dalle", "flux-ai", "ai-art", "ai-portrait", "generated-image",
]


# ==========================================================
# Helpers
# ==========================================================
def get_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""


def has_allowed_extension(url: str) -> bool:
    try:
        path = urlparse(url).path.lower()
        return path.endswith(ALLOWED_EXTENSIONS)
    except Exception:
        return False


def classify_real_trust(domain: str):
    """Return 'tier1', 'tier2', or None (blocked) for a REAL candidate domain."""
    if not domain:
        return None
    if domain in REAL_BLOCKED_DOMAINS:
        return None
    if domain in REAL_TIER1_DOMAINS or any(domain.endswith(s) for s in REAL_TIER1_SUFFIXES):
        return "tier1"
    return "tier2" if ALLOW_TIER2_REAL else None


def is_fake_domain_allowed(domain: str) -> bool:
    """Reject known trusted-editorial / stock-photo domains as sources of 'Fake' images."""
    if not domain:
        return False
    if domain in FAKE_EXCLUDED_DOMAINS:
        return False
    if any(domain.endswith(s) for s in REAL_TIER1_SUFFIXES):
        return False
    return True


def score_fake_candidate(title: str, image_url: str, source_url: str, domain: str) -> float:
    """
    A. Title check      -- strongest signal, weighted W_TITLE per matched keyword
    B. Image URL check   -- weighted W_URL per matched keyword
    C. Source page URL   -- weighted W_SOURCE_URL per matched keyword
    D. Domain whitelist  -- flat bonus W_WHITELIST if domain is a known AI-art host
    (Domain blacklist is handled earlier via is_fake_domain_allowed -- candidates
    from excluded domains never reach this function.)
    """
    score = 0.0
    title_l = (title or "").lower()
    image_url_l = (image_url or "").lower()
    source_url_l = (source_url or "").lower()

    if any(kw in title_l for kw in AI_TITLE_KEYWORDS):
        score += W_TITLE

    if any(kw in image_url_l for kw in AI_URL_KEYWORDS):
        score += W_URL

    if any(kw in source_url_l for kw in AI_URL_KEYWORDS):
        score += W_SOURCE_URL

    if domain in FAKE_WHITELIST_DOMAINS:
        score += W_WHITELIST

    return score


def load_completed_query_ids() -> set:
    """Query_IDs with a terminal status. 'error' is excluded so it retries."""
    if not SEARCHED_QUERIES_CSV.exists():
        return set()
    try:
        df = pd.read_csv(SEARCHED_QUERIES_CSV, dtype=str)
    except pd.errors.EmptyDataError:
        return set()
    if df.empty or "Status" not in df.columns:
        return set()
    done_mask = df["Status"].isin(DONE_STATUSES)
    return set(df.loc[done_mask, "Query_ID"].astype(str))


def load_existing_counts() -> dict:
    """(Identity_ID, Image_Type) -> number of already-accepted candidates."""
    counts = defaultdict(int)
    if not CANDIDATE_URLS_CSV.exists():
        return counts
    try:
        df = pd.read_csv(CANDIDATE_URLS_CSV, dtype=str)
    except pd.errors.EmptyDataError:
        return counts
    if df.empty:
        return counts
    grouped = df.groupby(["Identity_ID", "Image_Type"]).size()
    for (identity_id, image_type), n in grouped.items():
        counts[(str(identity_id), str(image_type))] = int(n)
    return counts


def append_candidates(rows: list) -> None:
    file_exists = CANDIDATE_URLS_CSV.exists()
    with open(CANDIDATE_URLS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def append_searched(record: dict) -> None:
    file_exists = SEARCHED_QUERIES_CSV.exists()
    with open(SEARCHED_QUERIES_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SEARCHED_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def sanitize_query(q: str) -> str:
    """Serper's free tier rejects quoted exact-phrase queries (and likely
    other 'advanced operator' syntax) with 'Query pattern not allowed for
    free accounts'. Strip quote characters before sending. The ORIGINAL
    text (with quotes) is still what gets stored in Search_Query for your
    records -- only the API request itself is sanitized."""
    return q.replace('"', "").replace("'", "").strip()


def target_for(image_type: str) -> int:
    return REAL_TARGET if str(image_type).strip().lower() == "real" else FAKE_TARGET


def build_identity_report() -> None:
    """Recompute per-identity Real/Fake counts and a keep/discard verdict.
    Safe to call anytime; does not touch the API or the ledger."""
    if not CANDIDATE_URLS_CSV.exists():
        logger.info("No candidate_urls.csv yet -- skipping report.")
        return
    try:
        df = pd.read_csv(CANDIDATE_URLS_CSV, dtype=str)
    except pd.errors.EmptyDataError:
        logger.info("candidate_urls.csv is empty -- skipping report.")
        return
    if df.empty:
        return

    df["_is_real"] = df["Image_Type"].str.strip().str.lower().eq("real")
    df["_is_fake"] = ~df["_is_real"]

    grouped = df.groupby(["Identity_ID", "Name"]).agg(
        Real_Count=("_is_real", "sum"),
        Fake_Count=("_is_fake", "sum"),
    ).reset_index()

    def verdict(row):
        if row["Real_Count"] > 0 and row["Fake_Count"] > 0:
            return "PAIRED"
        if row["Real_Count"] == 0 and row["Fake_Count"] == 0:
            return "DISCARD_NEITHER_FOUND"
        if row["Real_Count"] == 0:
            return "DISCARD_NO_REAL"
        return "DISCARD_NO_FAKE"

    grouped["Verdict"] = grouped.apply(verdict, axis=1)
    grouped = grouped.sort_values(["Verdict", "Identity_ID"]).reset_index(drop=True)
    grouped.to_csv(IDENTITY_REPORT_CSV, index=False, encoding="utf-8")

    n_paired = (grouped["Verdict"] == "PAIRED").sum()
    n_total = len(grouped)
    logger.info(f"Identity report: {n_paired}/{n_total} identities PAIRED -> {IDENTITY_REPORT_CSV}")
    print(f"Identity report saved: {n_paired}/{n_total} identities have both Real and Fake candidates.")
    print(f"  -> {IDENTITY_REPORT_CSV}")


# ==========================================================
# Main
# ==========================================================
def main() -> None:
    if not SEARCH_QUERY_CSV.exists():
        logger.error(f"Search query file not found: {SEARCH_QUERY_CSV}")
        print(f"ERROR: Search query file not found: {SEARCH_QUERY_CSV}")
        return

    queries = pd.read_csv(SEARCH_QUERY_CSV, dtype={"Query_ID": str, "Identity_ID": str})
    queries = queries.sort_values("Query_ID").reset_index(drop=True)  # deterministic order

    completed_ids = load_completed_query_ids()
    pending = queries[~queries["Query_ID"].isin(completed_ids)].reset_index(drop=True)

    batch = pending.head(MAX_QUERIES_PER_RUN) if MAX_QUERIES_PER_RUN is not None else pending

    logger.info("=" * 60)
    logger.info("STEP 3 START (Serper.dev Image Search)")
    logger.info(f"Total queries        : {len(queries)}")
    logger.info(f"Already completed    : {len(completed_ids)}")
    logger.info(f"Remaining pending    : {len(pending)}")
    logger.info(f"Queries this run     : {len(batch)}")
    logger.info(f"Real target/identity : {REAL_TARGET}")
    logger.info(f"Fake target/identity : {FAKE_TARGET}")
    logger.info(f"Fake score threshold : {FAKE_SCORE_THRESHOLD}")
    logger.info("=" * 60)

    if batch.empty:
        print("=" * 60)
        print("STEP 3 - Nothing to do")
        print("=" * 60)
        print(f"Total queries       : {len(queries)}")
        print(f"Already completed   : {len(completed_ids)}")
        print("All queries have already been searched or targets were met.")
        print("=" * 60)
        logger.info("No pending queries. Exiting.")
        build_identity_report()
        return

    counts = load_existing_counts()  # (Identity_ID, Image_Type) -> accepted count so far

    n_success = 0
    n_no_results = 0
    n_error = 0
    n_skipped_target = 0
    n_blocked = 0

    for _, row in tqdm(batch.iterrows(), total=len(batch)):
        query_id = str(row["Query_ID"])
        identity_id = str(row["Identity_ID"])
        image_type = str(row["Image_Type"])
        is_fake_query = image_type.strip().lower() != "real"
        key = (identity_id, image_type)
        target = target_for(image_type)

        # --- Early stop: identity already has enough of this type ---
        if counts[key] >= target:
            append_searched({
                "Query_ID": query_id,
                "Status": "skipped_target_met",
                "Search_Time": datetime.now().isoformat(timespec="seconds"),
                "Results_Returned": 0,
            })
            n_skipped_target += 1
            logger.info(f"{query_id} | Skipped - {identity_id}/{image_type} target already met "
                        f"({counts[key]}/{target})")
            continue  # no API call spent, no delay needed

        payload = {"q": sanitize_query(row["Query"]), "num": RESULTS_PER_QUERY}
        search_time = datetime.now().isoformat(timespec="seconds")

        try:
            r = requests.post(URL, headers=HEADERS, json=payload, timeout=30)

            if r.status_code != 200:
                body_lower = r.text.lower()
                if "query pattern not allowed" in body_lower:
                    logger.warning(f"{query_id} | BLOCKED (query syntax) | {r.text[:200]}")
                    append_searched({
                        "Query_ID": query_id, "Status": "blocked_query_syntax",
                        "Search_Time": search_time, "Results_Returned": 0,
                    })
                    n_blocked += 1
                    time.sleep(DELAY_SECONDS)
                    continue
                logger.warning(f"{query_id} | HTTP {r.status_code} | {r.text[:200]}")
                append_searched({
                    "Query_ID": query_id, "Status": "error",
                    "Search_Time": search_time, "Results_Returned": 0,
                })
                n_error += 1
                time.sleep(DELAY_SECONDS)
                continue

            data = r.json()
            items = data.get("images", [])

            if not items:
                logger.info(f"{query_id} | No results")
                append_searched({
                    "Query_ID": query_id, "Status": "no_results",
                    "Search_Time": search_time, "Results_Returned": 0,
                })
                n_no_results += 1
                time.sleep(DELAY_SECONDS)
                continue

            remaining_slots = target - counts[key]
            passed_filter = []  # candidates that cleared domain/extension rules

            for item in items:
                image_url = item.get("imageUrl", "")
                if not image_url or not has_allowed_extension(image_url):
                    continue

                domain = item.get("domain", "") or get_domain(item.get("link", "") or image_url)
                image_domain = get_domain(image_url)
                check_domain = domain or image_domain
                source_url = item.get("link", "")
                title = item.get("title", "")

                if not is_fake_query:
                    tier = classify_real_trust(check_domain)
                    if tier is None:
                        continue  # untrusted / blocked source, skip
                    score = None
                else:
                    if not is_fake_domain_allowed(check_domain):
                        continue  # excluded (looks like a trusted editorial/stock source)
                    tier = "n/a"
                    score = score_fake_candidate(title, image_url, source_url, check_domain)
                    if score < FAKE_SCORE_THRESHOLD:
                        continue  # doesn't look sufficiently AI-generated

                passed_filter.append({
                    "Query_ID": query_id,
                    "Identity_ID": identity_id,
                    "Name": row["Name"],
                    "Image_Type": image_type,
                    "Search_Query": row["Query"],
                    "Rank": item.get("position", 0),
                    "Title": title,
                    "Image_URL": image_url,
                    "Source_Page_URL": source_url,
                    "Display_Link": check_domain,
                    "Thumbnail_URL": item.get("thumbnailUrl", ""),
                    "Trust_Tier": tier,
                    "Score": score if score is not None else "",
                })

            # For fake queries, keep the highest-scoring candidates first when
            # there are more acceptable hits than open slots.
            if is_fake_query:
                passed_filter.sort(key=lambda c: c["Score"], reverse=True)
            IMAGES_PER_QUERY = 3

            candidate_rows = passed_filter[:IMAGES_PER_QUERY] if remaining_slots > 0 else []

            if not candidate_rows:
                logger.info(f"{query_id} | No results passed trust/score/extension filters")
                append_searched({
                    "Query_ID": query_id, "Status": "no_results",
                    "Search_Time": search_time, "Results_Returned": 0,
                })
                n_no_results += 1
                time.sleep(DELAY_SECONDS)
                continue

            append_candidates(candidate_rows)
            append_searched({
                "Query_ID": query_id, "Status": "success",
                "Search_Time": search_time, "Results_Returned": len(candidate_rows),
            })
            counts[key] += len(candidate_rows)

            n_success += 1
            logger.info(f"{query_id} | Success | {len(candidate_rows)} accepted "
                        f"({identity_id}/{image_type}: {counts[key]}/{target})")

        except Exception as e:
            logger.exception(f"{query_id} | Exception: {e}")
            append_searched({
                "Query_ID": query_id, "Status": "error",
                "Search_Time": search_time, "Results_Returned": 0,
            })
            n_error += 1

        time.sleep(DELAY_SECONDS)

    remaining_after = len(pending) - (n_success + n_no_results + n_skipped_target + n_blocked)

    logger.info("=" * 60)
    logger.info("STEP 3 RUN COMPLETE")
    logger.info(f"Success         : {n_success}")
    logger.info(f"No results      : {n_no_results}")
    logger.info(f"Skipped (target): {n_skipped_target}")
    logger.info(f"Blocked (syntax): {n_blocked} (terminal, will NOT retry)")
    logger.info(f"Errors          : {n_error} (will retry next run)")
    logger.info(f"Remaining       : {remaining_after}")
    logger.info("=" * 60)

    print("=" * 60)
    print("STEP 3 RUN COMPLETE")
    print("=" * 60)
    print(f"Total queries (all time)   : {len(queries)}")
    print(f"Processed this run         : {len(batch)}")
    print(f"  Success                  : {n_success}")
    print(f"  No results               : {n_no_results}")
    print(f"  Skipped (target met)     : {n_skipped_target}  <- saved API calls")
    print(f"  Blocked (query syntax)   : {n_blocked}  <- terminal, won't retry")
    print(f"  Errors (will retry)      : {n_error}")
    print(f"Remaining pending queries  : {remaining_after}")
    print(f"Candidate URLs file        : {CANDIDATE_URLS_CSV}")
    print(f"Searched log file          : {SEARCHED_QUERIES_CSV}")
    print(f"Log file                   : {LOG_FILE}")
    print("=" * 60)
    if remaining_after > 0:
        next_batch = min(remaining_after, MAX_QUERIES_PER_RUN) if MAX_QUERIES_PER_RUN else remaining_after
        print(f"Run the script again to continue with the next batch ({next_batch} queries).")

    build_identity_report()


if __name__ == "__main__":
    try:
        if "--report-only" in sys.argv:
            build_identity_report()
        else:
            main()
    except KeyboardInterrupt:
        logger.warning("Run interrupted by user (KeyboardInterrupt). "
                        "Progress up to the last completed query has been saved. "
                        "Re-run the script to resume.")
        print("\nInterrupted. Progress saved - re-run the script to resume.")