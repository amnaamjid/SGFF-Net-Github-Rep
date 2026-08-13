"""
build_master_sheet.py

Combines the 'Name' column from every .xlsx file in Source_Data/
into a single Master Sheet, tagging each row with its source file.

No deduplication or name-cleaning happens here — that's a separate
step. This just pools everything together.

Usage:
    python build_master_sheet.py
"""

import os
import glob
import pandas as pd

# ---------- CONFIG ----------
SOURCE_DIR = "Source_Data"
OUTPUT_FILE = "Master_Sheet_All.xlsx"


# Column names that should be treated as the "Name" column,
# in case a file doesn't literally use "Name"
NAME_COLUMN_ALIASES = [
    "name", "full name", "athlete name", "player name",
    "person name", "presenter name", "author name",
    "scientist name", "doctor name", "ceo name",
]
# -----------------------------


def find_name_column(columns):
    """Return the actual column name that matches a known 'name' alias."""
    normalized = {c: str(c).strip().lower() for c in columns}
    for original, norm in normalized.items():
        if norm in NAME_COLUMN_ALIASES:
            return original
    # fallback: any column whose header just contains 'name'
    for original, norm in normalized.items():
        if "name" in norm:
            return original
    return None


def main():
    files = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.xlsx")))
    files = [f for f in files if os.path.basename(f) not in SKIP_FILES
             and ":Zone.Identifier" not in f]

    if not files:
        print(f"No .xlsx files found in '{SOURCE_DIR}/'.")
        return

    all_rows = []
    skipped_files = []

    for filepath in files:
        filename = os.path.basename(filepath)
        try:
            xls = pd.ExcelFile(filepath)
        except Exception as e:
            print(f"[SKIP] Could not open '{filename}': {e}")
            skipped_files.append(filename)
            continue

        file_had_names = False

        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name)
            except Exception as e:
                print(f"[SKIP] Could not read sheet '{sheet_name}' in '{filename}': {e}")
                continue

            if df.empty or len(df.columns) == 0:
                continue

            name_col = find_name_column(df.columns)
            if name_col is None:
                continue

            names = df[name_col].dropna().astype(str).str.strip()
            names = names[names != ""]

            if names.empty:
                continue

            file_had_names = True
            for name in names:
                all_rows.append({
                    "Name": name,
                    "Source_File": filename,
                    "Source_Sheet": sheet_name,
                    "Source_Column": name_col,
                })

        if not file_had_names:
            print(f"[WARNING] No Name-like column found in '{filename}' — check it manually.")
            skipped_files.append(filename)

    if not all_rows:
        print("No names were extracted from any file. Aborting.")
        return

    master_df = pd.DataFrame(all_rows)
    master_df.insert(0, "ID", range(1, len(master_df) + 1))

    master_df.to_excel(OUTPUT_FILE, index=False)

    print("\n--- DONE ---")
    print(f"Total rows (identities, with duplicates across sources): {len(master_df)}")
    print(f"Files processed: {len(files) - len(skipped_files)}")
    if skipped_files:
        print(f"Files with issues ({len(skipped_files)}): {skipped_files}")
    print(f"Master sheet saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
