from pathlib import Path
import pandas as pd
import yaml
import json
import logging

# ============================================================
# Project Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CONFIG = ROOT / "Config" / "config.yaml"
QUERY_TEMPLATE = ROOT / "Config" / "query_templates.yaml"

with open(CONFIG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

QUEUE = ROOT / cfg["queue_csv"]

SEARCH_DIR = ROOT / "Data" / "Search"
SEARCH_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = SEARCH_DIR / "search_queries.csv"
REPORT = ROOT / "Data" / "Reports" / "search_query_report.json"
LOG = ROOT / "Data" / "Logs" / "02_generate_search_queries.log"

logging.basicConfig(
    filename=LOG,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filemode="w"
)

logging.info("STEP 2 STARTED")

# ============================================================
# Load queue
# ============================================================

queue = pd.read_csv(QUEUE)

with open(QUERY_TEMPLATE, "r", encoding="utf-8") as f:
    templates = yaml.safe_load(f)

records = []

query_id = 1

# ============================================================
# Generate Queries
# ============================================================

for _, row in queue.iterrows():

    identity_id = row["Identity_ID"]
    name = row["Name"]

    # ---------- REAL ----------

    for priority, template in enumerate(templates["real"], start=1):

        records.append({

            "Query_ID": f"Q{query_id:06d}",

            "Identity_ID": identity_id,

            "Name": name,

            "Image_Type": "Real",

            "Priority": priority,

            "Query": template.format(name=name)

        })

        query_id += 1

    # ---------- FAKE ----------

    for priority, template in enumerate(templates["fake"], start=1):

        records.append({

            "Query_ID": f"Q{query_id:06d}",

            "Identity_ID": identity_id,

            "Name": name,

            "Image_Type": "Fake",

            "Priority": priority,

            "Query": template.format(name=name)

        })

        query_id += 1

# ============================================================
# Save
# ============================================================

df = pd.DataFrame(records)

df.to_csv(OUTPUT, index=False, encoding="utf-8")

report = {

    "Identities": int(queue.shape[0]),

    "Real Queries": len(templates["real"]),

    "Fake Queries": len(templates["fake"]),

    "Total Queries": int(df.shape[0])

}

with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

logging.info(report)
logging.info("STEP 2 FINISHED")

print("=" * 60)
print("STEP 2 COMPLETED")
print("=" * 60)
print(f"Identities     : {queue.shape[0]}")
print(f"Total Queries  : {df.shape[0]}")
print(f"Saved To       : {OUTPUT}")
print("=" * 60)
