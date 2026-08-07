#!/usr/bin/env python3
"""
resolve_leakage.py
====================
Finds identity overlap (same as run_full_leakage_check.py) but goes one step
further: for every overlap found, it tells you the EXACT FILENAMES to remove
so the datasets become fully identity-disjoint.

RULE USED TO DECIDE WHAT GETS REMOVED
--------------------------------------
For every required pair, one side is "trained on" and the other is "evaluated
on". We always remove the overlapping identities from the TRAINED-ON side,
never from the evaluation side -- evaluation sets (DiffFace, DFFD_A test,
DFFD_B) should stay fixed/untouched so results stay comparable across papers
and across your own experiments. Concretely:

    dff_c_train   vs diffface      -> remove from dff_c_train
    dff_c_val     vs diffface      -> remove from dff_c_val
    dff_c_train   vs dffd_a(all)   -> remove from dff_c_train
    dff_c_val     vs dffd_a(all)   -> remove from dff_c_val
    dff_c_train   vs dffd_b        -> remove from dff_c_train
    dff_c_val     vs dffd_b        -> remove from dff_c_val
    dffd_a_train  vs dffd_b        -> remove from dffd_a_train
    dffd_a_val    vs dffd_b        -> remove from dffd_a_val

OUTPUT
------
For each of dff_c_train, dff_c_val, dffd_a_train, dffd_a_val, it writes:

    removed_rows_<pool>.csv     <- exact filenames + identity to delete
    cleaned_<original_filename> <- the original metadata file with those
                                    rows dropped (same format, csv/xlsx),
                                    ready to replace your current file

USAGE
-----
pip install pandas openpyxl --break-system-packages

python resolve_leakage.py --dir /home/amna/Ready_to_send_Dataset --out /home/amna/Ready_to_send_Dataset/cleaned
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("Missing dependency. Install with:\n  pip install pandas openpyxl --break-system-packages")

FILES = {
    "dff_c_train":  "DFF_C_Train_Real_Selected_Metadata.csv",
    "dff_c_val":    "DFF_C_Val_Real_Selected_Metadata.csv",
    "dff_c_test":   "DFF_C_Test_Real_Selected_Meteadata.csv",
    "dffd_a_train": "DFFD_A_Train_Real_Selected_Metadata.xlsx",
    "dffd_a_val":   "DFFD_A_Val_Real_Selected_Metadata.xlsx",
    "dffd_a_test":  "DFFD_A_Test_Real_Selected_Metadata.xlsx",
    "dffd_b":       "DFFD_B_Real_Selected_Metadata.xlsx",
    "diffface":     "DiffFace_Real_Selected_Metadata.xlsx",
}

NAME_COLUMN_CANDIDATES = ["identity_name", "identity", "name", "person", "person_name"]
FILE_COLUMN_CANDIDATES = ["filename", "file", "image", "image_name", "file_name", "image_path"]

# (trained_on_pool, evaluated_on_pool) -- overlap always removed from trained_on_pool
REQUIRED_PAIRS = [
    ("dff_c_train", "diffface"),
    ("dff_c_val", "diffface"),
    ("dff_c_train", "dffd_a_all"),
    ("dff_c_val", "dffd_a_all"),
    ("dff_c_train", "dffd_b"),
    ("dff_c_val", "dffd_b"),
    ("dffd_a_train", "dffd_b"),
    ("dffd_a_val", "dffd_b"),
]


def normalize_name(name) -> str:
    if pd.isna(name):
        return ""
    name = str(name)
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.strip().lower()
    name = re.sub(r"[\s_]+", "_", name)
    name = re.sub(r"[<>:\"/\\|?*]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def find_column(df, candidates, label, filepath, required=True):
    lower_map = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    if required:
        sys.exit(
            f"ERROR: could not find a '{label}' column in {filepath}\n"
            f"Columns found: {list(df.columns)}\nExpected one of: {candidates}"
        )
    return None


def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"ERROR: file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    sys.exit(f"ERROR: unsupported file type: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="Folder to write cleaned files + removal lists")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading files...\n")
    dfs, name_cols, file_cols = {}, {}, {}
    for pool_name, filename in FILES.items():
        path = args.dir / filename
        df = load_df(path)
        name_col = find_column(df, NAME_COLUMN_CANDIDATES, "identity", path)
        file_col = find_column(df, FILE_COLUMN_CANDIDATES, "filename", path, required=False)
        df["_identity_norm"] = df[name_col].apply(normalize_name)
        dfs[pool_name] = df
        name_cols[pool_name] = name_col
        file_cols[pool_name] = file_col
        print(f"  {pool_name:<14}: {len(df):>6,} rows  "
              f"(identity col='{name_col}', filename col={file_col!r})  [{filename}]")
        if file_col is None:
            print(f"    WARNING: no filename column detected for {pool_name} -- "
                  f"can't list exact files to remove for this pool, only identities.")

    def pool_ids(pool_name):
        return set(dfs[pool_name]["_identity_norm"]) - {""}

    dffd_a_all_ids = pool_ids("dffd_a_train") | pool_ids("dffd_a_val") | pool_ids("dffd_a_test")
    pool_id_sets = {p: pool_ids(p) for p in FILES}
    pool_id_sets["dffd_a_all"] = dffd_a_all_ids

    # accumulate identities to remove, per trained-on pool
    to_remove_ids = {"dff_c_train": set(), "dff_c_val": set(),
                      "dffd_a_train": set(), "dffd_a_val": set()}

    print("\n" + "=" * 70)
    print("OVERLAP CHECK")
    print("=" * 70)
    for trained_pool, eval_pool in REQUIRED_PAIRS:
        overlap = pool_id_sets[trained_pool] & pool_id_sets[eval_pool]
        status = "PASS - disjoint" if not overlap else f"FAIL - {len(overlap)} shared identities -> will remove from {trained_pool}"
        print(f"{trained_pool:<14} vs {eval_pool:<14}: {status}")
        if overlap:
            to_remove_ids[trained_pool] |= overlap

    print("\n" + "=" * 70)
    print("WRITING REMOVAL LISTS + CLEANED FILES")
    print("=" * 70)
    any_removed = False
    for pool_name, ids_to_drop in to_remove_ids.items():
        df = dfs[pool_name]
        file_col = file_cols[pool_name]
        name_col = name_cols[pool_name]
        mask = df["_identity_norm"].isin(ids_to_drop)
        removed_rows = df[mask]
        kept_rows = df[~mask].drop(columns=["_identity_norm"])

        if len(removed_rows) == 0:
            print(f"  {pool_name}: nothing to remove.")
            continue

        any_removed = True
        removal_cols = [c for c in [file_col, name_col] if c is not None]
        removal_path = args.out / f"removed_rows_{pool_name}.csv"
        removed_rows[removal_cols].to_csv(removal_path, index=False)

        original_filename = FILES[pool_name]
        cleaned_path = args.out / f"cleaned_{original_filename}"
        if cleaned_path.suffix.lower() == ".csv":
            kept_rows.to_csv(cleaned_path, index=False)
        else:
            kept_rows.to_excel(cleaned_path, index=False)

        print(f"  {pool_name}: removing {len(removed_rows):,} rows "
              f"({len(ids_to_drop):,} identities)")
        print(f"    -> filenames to delete listed in : {removal_path}")
        print(f"    -> cleaned replacement file       : {cleaned_path}")

    print("\n" + "=" * 70)
    if any_removed:
        print("DONE. Review the removed_rows_*.csv files, then physically delete those")
        print("image files from disk and replace your metadata files with the cleaned_*")
        print("versions. Re-run run_full_leakage_check.py afterward to confirm PASS.")
    else:
        print("DONE. No overlap found anywhere -- datasets are already identity-disjoint,")
        print("nothing needs to be removed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
