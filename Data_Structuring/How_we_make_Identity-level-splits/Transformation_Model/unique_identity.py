#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path("/home/amna/Ready_to_send_Dataset/DFFD")

INPUT = ROOT / "master_identity_filtered.csv"
OUTPUT = ROOT / "Metadata" / "unique_identities.csv"

# --------------------------------------------------

print("Loading metadata...")

df = pd.read_csv(INPUT)

# Get unique identity names
identities = (
    df["identity"]
    .dropna()
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

# Save as CSV
out_df = pd.DataFrame({
    "identity": identities
})

out_df.to_csv(OUTPUT, index=False)

print(f"Total unique identities : {len(out_df):,}")
print(f"Saved : {OUTPUT}")
