import os
import re
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

MASTER_CSV = "/home/amna/Dataset/DFFD/master_identity_filtered.csv"

FAKE_ROOT = "/home/amna/Dataset/DFFD/Fake"

OUTPUT_DIR = "/home/amna/Dataset/DFFD/Metadata"

# ============================================================

output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Load identities
# ------------------------------------------------------------

print("Loading master identities...")

df = pd.read_csv(MASTER_CSV)

identities = sorted(df["identity"].unique())

print(f"Total identities : {len(identities):,}")

# ------------------------------------------------------------
# Fake folders
# ------------------------------------------------------------

folders = {
    "pggan_v1": "pggan_v1",
    "pggan_v2": "pggan_v2",
    "stylegan": "stylegan",
    "stargan": "stargan",
}

# ------------------------------------------------------------
# Build identity index
# ------------------------------------------------------------

print("\nScanning fake folders...")

coverage = {}

for key, folder in folders.items():

    path = Path(FAKE_ROOT) / folder

    identity_set = set()

    count = 0

    for file in path.iterdir():

        if not file.is_file():
            continue

        name = file.stem

        # Split before "_F_"
        if "_F_" not in name:
            continue

        identity = name.split("_F_")[0]

        identity_set.add(identity)

        count += 1

    coverage[key] = identity_set

    print(f"{folder:12s} : {count:,} files   {len(identity_set):,} identities")

# ------------------------------------------------------------
# Verification
# ------------------------------------------------------------

rows = []

for identity in identities:

    pg1 = identity in coverage["pggan_v1"]
    pg2 = identity in coverage["pggan_v2"]
    sty = identity in coverage["stylegan"]
    stg = identity in coverage["stargan"]

    rows.append({
        "identity": identity,
        "pggan_v1": pg1,
        "pggan_v2": pg2,
        "stylegan": sty,
        "stargan": stg,
        "complete": pg1 and pg2 and sty and stg
    })

coverage_df = pd.DataFrame(rows)

coverage_df.to_csv(
    output_dir / "fake_identity_coverage.csv",
    index=False
)

missing = coverage_df[coverage_df["complete"] == False]

missing.to_csv(
    output_dir / "missing_fake_report.csv",
    index=False
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n==============================")

print(f"Complete identities : {coverage_df['complete'].sum():,}")

print(f"Incomplete identities : {len(missing):,}")

print("\nAvailability")

for col in ["pggan_v1", "pggan_v2", "stylegan", "stargan"]:

    print(f"{col:12s}: {coverage_df[col].sum():,}")

print("\nSaved")

print(output_dir / "fake_identity_coverage.csv")

print(output_dir / "missing_fake_report.csv")
