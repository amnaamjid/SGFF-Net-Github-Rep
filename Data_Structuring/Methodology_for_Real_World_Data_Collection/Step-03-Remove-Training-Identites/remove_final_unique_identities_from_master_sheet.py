#!/usr/bin/env python3
"""
Remove identities already present in final_unique_identities.csv
from Master_Sheet.xlsx.

Author: Amna
"""

from pathlib import Path
import pandas as pd
import re

# ==========================================================
# Paths
# ==========================================================

ROOT = Path("/home/amna/In-The-Wild-Dataset-Collection")

MASTER_FILE = ROOT / "Master_Sheet_Unique_Clean.xlsx"
IDENTITY_FILE = ROOT / "final_unique_identities.csv"

OUTPUT_MASTER = ROOT / "Master_Sheet_Unique.xlsx"
OUTPUT_REMOVED = ROOT / "Master_Sheet_Removed.xlsx"
OUTPUT_DUPLICATES = ROOT / "Master_Sheet_DuplicateRows.xlsx"

# ==========================================================
# Name Normalization
# ==========================================================

def normalize_name(name):

    if pd.isna(name):
        return ""

    name = str(name).strip()

    # Remove quotes
    name = name.replace('"', '')

    # Underscore -> space
    name = name.replace("_", " ")

    # A. J. -> AJ
    name = re.sub(r'\b([A-Z])\.\s*([A-Z])\.', r'\1\2', name)
    name = re.sub(r'\b([A-Z])\.\s*([A-Z])\b', r'\1\2', name)

    # Remove remaining dots
    name = name.replace(".", "")

    # Remove multiple spaces
    name = re.sub(r"\s+", " ", name)

    return name.strip()


def make_key(name):

    key = normalize_name(name).lower()

    key = re.sub(r"[^a-z0-9]", "", key)

    return key


# ==========================================================
# Load Data
# ==========================================================

print("=" * 60)
print("Loading files...")
print("=" * 60)

master = pd.read_excel(MASTER_FILE)
identities = pd.read_csv(IDENTITY_FILE)

# ----------------------------------------------------------
# Verify columns
# ----------------------------------------------------------

if "Name" not in master.columns:
    raise ValueError("Master sheet must contain column 'Name'.")

if "merge_key" not in identities.columns:
    raise ValueError(
        "final_unique_identities.csv must contain column 'merge_key'."
    )

print(f"Master Sheet rows      : {len(master):,}")
print(f"Known identities       : {len(identities):,}")

# ==========================================================
# Create Merge Keys
# ==========================================================

master["standardized_name"] = master["Name"].apply(normalize_name)
master["merge_key"] = master["Name"].apply(make_key)

known_keys = set(
    identities["merge_key"].astype(str)
)

# ==========================================================
# Detect duplicates INSIDE Master Sheet
# ==========================================================

duplicate_rows = master[
    master.duplicated("merge_key", keep="first")
].copy()

master_unique = master.drop_duplicates(
    subset="merge_key",
    keep="first"
)

print(f"Duplicate rows in Master Sheet : {len(duplicate_rows):,}")

# ==========================================================
# Remove existing identities
# ==========================================================

removed = master_unique[
    master_unique["merge_key"].isin(known_keys)
].copy()

remaining = master_unique[
    ~master_unique["merge_key"].isin(known_keys)
].copy()

# ==========================================================
# Sort
# ==========================================================

remaining = remaining.sort_values(
    "standardized_name"
).reset_index(drop=True)

removed = removed.sort_values(
    "standardized_name"
).reset_index(drop=True)

duplicate_rows = duplicate_rows.sort_values(
    "standardized_name"
).reset_index(drop=True)

# ==========================================================
# Reset Serial Numbers
# ==========================================================

remaining["S.No"] = range(1, len(remaining) + 1)

# ==========================================================
# Drop helper columns
# ==========================================================

helper_cols = ["standardized_name", "merge_key"]

remaining = remaining.drop(columns=helper_cols)
removed = removed.drop(columns=helper_cols)
duplicate_rows = duplicate_rows.drop(columns=helper_cols)

# ==========================================================
# Save
# ==========================================================

remaining.to_excel(
    OUTPUT_MASTER,
    index=False
)

removed.to_excel(
    OUTPUT_REMOVED,
    index=False
)

duplicate_rows.to_excel(
    OUTPUT_DUPLICATES,
    index=False
)

# ==========================================================
# Report
# ==========================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Original Master Sheet        : {len(master):,}")
print(f"Internal Duplicate Rows      : {len(duplicate_rows):,}")
print(f"Unique Master Identities     : {len(master_unique):,}")
print(f"Already Present (Removed)    : {len(removed):,}")
print(f"Final New Identities         : {len(remaining):,}")

print("\nFiles Saved")
print("-" * 60)
print(OUTPUT_MASTER)
print(OUTPUT_REMOVED)
print(OUTPUT_DUPLICATES)

print("\nDone.")
