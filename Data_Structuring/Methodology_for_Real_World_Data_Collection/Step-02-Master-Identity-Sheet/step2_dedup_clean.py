"""
step2_dedup_clean.py

STEP 2 of the pipeline.

Input:  Master_Sheet_All.xlsx   (raw pooled names from Source_Data, built in Step 1)
Output: Master_Sheet_Unique_Clean.xlsx  (deduplicated, cleaned, one row per identity)

Dedup logic:
  - Names are matched using a "merge_key": lowercase, all non-alphanumeric
    characters stripped (spaces, hyphens, periods, apostrophes, etc.)
    e.g. "A-1" -> "a1", "AJ Allmendinger" -> "ajallmendinger"
  - This is the SAME key format used in your existing final_unique_identities.csv,
    so Step 3 can match against it directly.
  - First occurrence of each merge_key is kept as the canonical Name.

Usage:
    python step2_dedup_clean.py
"""

import re
import pandas as pd

# ---------- CONFIG ----------
INPUT_FILE = "Master_Sheet_All.xlsx"
OUTPUT_FILE = "Master_Sheet_Unique_Clean.xlsx"
# -----------------------------


def make_merge_key(name):
    """lowercase + strip all non-alphanumeric characters"""
    if not isinstance(name, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def main():
    df = pd.read_excel(INPUT_FILE)
    if "Name" not in df.columns:
        raise ValueError(f"'{INPUT_FILE}' has no 'Name' column. Found: {list(df.columns)}")

    df["Name"] = df["Name"].astype(str).str.strip()
    df = df[df["Name"] != ""]
    df["merge_key"] = df["Name"].apply(make_merge_key)
    df = df[df["merge_key"] != ""]

    total_raw = len(df)

    # Keep first occurrence of each merge_key as canonical name.
    # Also collect which source files each identity appeared in.
    if "Source_File" in df.columns:
        sources = (
            df.groupby("merge_key")["Source_File"]
            .apply(lambda s: ", ".join(sorted(set(s))))
        )
    else:
        sources = None

    deduped = df.drop_duplicates(subset="merge_key", keep="first").copy()
    deduped = deduped.reset_index(drop=True)

    if sources is not None:
        deduped["Sources"] = deduped["merge_key"].map(sources)

    out_cols = ["Name", "merge_key"] + (["Sources"] if sources is not None else [])
    final_df = deduped[out_cols].reset_index(drop=True)
    final_df.insert(0, "ID", range(1, len(final_df) + 1))

    final_df.to_excel(OUTPUT_FILE, index=False)

    print("\n--- STEP 2 DONE ---")
    print(f"Raw rows in Master_Sheet_All:     {total_raw}")
    print(f"Unique identities after dedup:    {len(final_df)}")
    print(f"Duplicates removed:               {total_raw - len(final_df)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
