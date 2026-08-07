from pathlib import Path
import pandas as pd
import yaml
import json
import logging
import unicodedata

# ==========================================================
# Paths
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

CONFIG = ROOT / "Config" / "config.yaml"

with open(CONFIG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

MASTER = ROOT / cfg["master_sheet"]
QUEUE = ROOT / cfg["queue_csv"]
DUPLICATES = ROOT / cfg["duplicate_csv"]
INVALID = ROOT / cfg["invalid_csv"]
REPORT = ROOT / cfg["report_json"]
LOG = ROOT / cfg["log_file"]

# ==========================================================
# Create folders
# ==========================================================

for p in [QUEUE, DUPLICATES, INVALID, REPORT, LOG]:
    p.parent.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Logging
# ==========================================================

logging.basicConfig(
    filename=LOG,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filemode="w"
)

logging.info("STEP 1 STARTED")

# ==========================================================
# Check files
# ==========================================================

if not MASTER.exists():
    raise FileNotFoundError(f"Master sheet not found:\n{MASTER}")

# ==========================================================
# Read Excel
# ==========================================================

df = pd.read_excel(MASTER)

required = ["Name"]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Required column missing: {col}")

optional = ["Profession", "Gender"]

for col in optional:
    if col not in df.columns:
        df[col] = ""

# ==========================================================
# Name cleaning
# ==========================================================

def normalize_name(name):

    if pd.isna(name):
        return None

    name = str(name)

    name = unicodedata.normalize("NFC", name)

    name = " ".join(name.split())

    name = name.strip()

    if name == "":
        return None

    if name.lower() in {"nan", "none", "unknown"}:
        return None

    return name

# ==========================================================
# Build queue
# ==========================================================

seen = {}

queue = []

duplicates = []

invalid = []

for excel_row, row in enumerate(df.itertuples(index=False), start=2):

    name = normalize_name(row.Name)

    if name is None:
        invalid.append({
            "Excel_Row": excel_row,
            "Reason": "Invalid Name"
        })
        continue

    key = name.casefold()

    if key in seen:
        duplicates.append({
            "Excel_Row": excel_row,
            "Duplicate_Name": name,
            "First_Appearance_Row": seen[key]
        })
        continue

    seen[key] = excel_row

    queue.append({

        "Identity_ID": f"ID{len(queue)+1:06d}",

        "Order": len(queue)+1,

        "Name": name,

        "Profession": "" if pd.isna(row.Profession) else str(row.Profession),

        "Gender": "" if pd.isna(row.Gender) else str(row.Gender),

        "Search_Status": "Pending",

        "Download_Status": "Pending",

        "Verification_Status": "Pending",

        "Final_Status": "Pending",

        "Real_Count": 0,

        "Fake_Count": 0

    })

queue_df = pd.DataFrame(queue)

dup_df = pd.DataFrame(duplicates)

invalid_df = pd.DataFrame(invalid)

queue_df.to_csv(QUEUE, index=False)

dup_df.to_csv(DUPLICATES, index=False)

invalid_df.to_csv(INVALID, index=False)

report = {

    "Initial_Rows": int(len(df)),
    "Valid_Identities": int(len(queue_df)),
    "Duplicate_Names": int(len(dup_df)),
    "Invalid_Rows": int(len(invalid_df))

}

with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

logging.info(report)
logging.info("STEP 1 FINISHED")

print("\n========== STEP 1 COMPLETE ==========\n")
print(f"Initial Rows      : {len(df)}")
print(f"Valid Identities  : {len(queue_df)}")
print(f"Duplicate Names   : {len(dup_df)}")
print(f"Invalid Rows      : {len(invalid_df)}")
print("\nQueue created successfully.\n")