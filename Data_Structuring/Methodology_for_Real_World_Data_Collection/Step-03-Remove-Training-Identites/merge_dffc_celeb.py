#!/usr/bin/env python3
"""
Merge DFFC and Celeb identities into one standardized master list.

Author: Amna
"""

from pathlib import Path
import pandas as pd
import re

# ==========================================================
# Paths
# ==========================================================

ROOT = Path("/home/amna/In-The-Wild-Dataset-Collection")

DFFC_FILE = ROOT / "dffc_unique_identities.csv"
CELEB_FILE = ROOT / "celeb_unique_identities.csv"

OUT_DFFC = ROOT / "normalized_dffc.csv"
OUT_CELEB = ROOT / "normalized_celeb.csv"
OUT_FINAL = ROOT / "final_unique_identities.csv"

# ==========================================================
# Name normalization
# ==========================================================

def normalize_name(name: str) -> str:
    """Convert names into one readable standard."""

    if pd.isna(name):
        return ""

    name = str(name).strip()

    # remove quotes
    name = name.replace('"', '')
    name = name.replace("'", "'")

    # underscores -> spaces
    name = name.replace("_", " ")

    # collapse initials
    # A. J.  -> AJ
    # A.J.   -> AJ
    name = re.sub(r'\b([A-Z])\.\s*([A-Z])\.', r'\1\2', name)
    name = re.sub(r'\b([A-Z])\.\s*([A-Z])\b', r'\1\2', name)

    # remove remaining dots
    name = name.replace(".", "")

    # remove extra spaces
    name = re.sub(r"\s+", " ", name)

    name = name.strip()

    return name


def make_key(name: str) -> str:
    """Create duplicate matching key."""

    key = normalize_name(name).lower()

    # remove everything except letters and numbers
    key = re.sub(r"[^a-z0-9]", "", key)

    return key


# ==========================================================
# Load DFFC
# ==========================================================

print("Loading DFFC...")

dffc = pd.read_csv(DFFC_FILE)

dffc = dffc[["identity"]].copy()

dffc["standardized_name"] = dffc["identity"].apply(normalize_name)
dffc["merge_key"] = dffc["identity"].apply(make_key)
dffc["source"] = "DFFC"

dffc = dffc.drop_duplicates("merge_key")

dffc.to_csv(OUT_DFFC, index=False)

print(f"DFFC identities: {len(dffc):,}")

# ==========================================================
# Load Celeb
# ==========================================================

print("Loading Celeb...")

celeb = pd.read_csv(CELEB_FILE)

celeb["standardized_name"] = celeb["identity"].apply(normalize_name)
celeb["merge_key"] = celeb["identity"].apply(make_key)
celeb["source"] = "Celeb"

celeb = celeb.drop_duplicates("merge_key")

celeb.to_csv(OUT_CELEB, index=False)

print(f"Celeb identities: {len(celeb):,}")

# ==========================================================
# Merge
# ==========================================================

print("Merging...")

master = {}

# Add DFFC first (preferred display name)
for _, row in dffc.iterrows():

    master[row.merge_key] = {
        "standardized_name": row.standardized_name,
        "merge_key": row.merge_key,
        "source": "DFFC",
    }

# Merge Celeb
for _, row in celeb.iterrows():

    key = row.merge_key

    if key in master:

        master[key]["source"] = "Both"

    else:

        master[key] = {
            "standardized_name": row.standardized_name,
            "merge_key": key,
            "source": "Celeb",
        }

# ==========================================================
# Save
# ==========================================================

final = pd.DataFrame(master.values())

final = final.sort_values("standardized_name").reset_index(drop=True)

final.to_csv(OUT_FINAL, index=False)

print("\n==========================")
print("Finished")
print("==========================")

print(f"DFFC unique : {len(dffc):,}")
print(f"Celeb unique: {len(celeb):,}")
print(f"Final unique: {len(final):,}")

print("\nSources")
print(final["source"].value_counts())

print(f"\nSaved:")
print(OUT_DFFC)
print(OUT_CELEB)
print(OUT_FINAL)
