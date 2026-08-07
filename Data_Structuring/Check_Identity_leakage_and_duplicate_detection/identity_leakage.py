#!/usr/bin/env python3
"""
run_full_leakage_check.py
===========================
ONE script that does everything, built for this exact folder:

  ~/Ready_to_send_Dataset/
      DFF_C_Train_Real_Selected_Metadata.csv
      DFF_C_Val_Real_Selected_Metadata.csv
      DFF_C_Test_Real_Selected_Meteadata.csv        <- note: typo "Meteadata" in your filename
      DFFD_A_Train_Real_Selected_Metadata.xlsx
      DFFD_A_Val_Real_Selected_Metadata.xlsx
      DFFD_A_Test_Real_Selected_Metadata.xlsx
      DFFD_B_Real_Selected_Metadata.xlsx
      DiffFace_Real_Selected_Metadata.xlsx

WHAT IT DOES
------------
1. Reads each file (csv or xlsx, auto-detected).
2. Finds the identity-name column automatically (looks for a column named
   identity_name / identity / name / person / person_name -- case-insensitive).
3. Normalizes every identity name the same way your rename script did:
   strips accents, collapses whitespace/underscores, so "Diana Damrau" and
   "Diana_Damrau" and "Diana  Damrau" are all treated as the SAME identity.
   Without this step, near-identical name spellings would look like
   different people and hide real leakage.
4. Builds these identity pools:
     dff_c_train, dff_c_val, dff_c_test
     dffd_a_train, dffd_a_val, dffd_a_test
     dffd_b            (single pool, no train/val/test)
     diffface           (single pool, no train/val/test)
5. Runs every leakage check that matters for YOUR evaluation setup:
     DFF_C(train/val) is trained,  evaluated on DiffFace and on DFFD_A + DFFD_B
     DFFD_A(train/val) is trained, evaluated on DFFD_B
   So the required-disjoint pairs are:
     dff_c_train  vs diffface
     dff_c_val    vs diffface
     dff_c_train  vs dffd_a_test  + dffd_a_val + dffd_a_train + dffd_b
     dff_c_val    vs dffd_a_test  + dffd_a_val + dffd_a_train + dffd_b
     dffd_a_train vs dffd_b
     dffd_a_val   vs dffd_b
6. Prints a clear PASS/FAIL per line, writes a full text report, and gives a
   final verdict you can quote directly in the rebuttal.

USAGE
-----
pip install pandas openpyxl --break-system-packages

python run_full_leakage_check.py --dir /home/amna/Ready_to_send_Dataset --report leakage_report.txt

If any of your column names differ from what this script auto-detects, it
will tell you exactly which file/column it couldn't find -- fix the name in
the file, or tell me the real column name and I'll hardcode it.
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

# Exact filenames as they exist on disk right now.
FILES = {
    "dff_c_train":  "DFF_C_Train_Real_Selected_Metadata.csv",
    "dff_c_val":    "DFF_C_Val_Real_Selected_Metadata.csv",
    "dff_c_test":   "DFF_C_Test_Real_Selected_Meteadata.csv",  # typo kept intentionally
    "dffd_a_train": "DFFD_A_Train_Real_Selected_Metadata.xlsx",
    "dffd_a_val":   "DFFD_A_Val_Real_Selected_Metadata.xlsx",
    "dffd_a_test":  "DFFD_A_Test_Real_Selected_Metadata.xlsx",
    "dffd_b":       "DFFD_B_Real_Selected_Metadata.xlsx",
    "diffface":     "DiffFace_Real_Selected_Metadata.xlsx",
}

NAME_COLUMN_CANDIDATES = [
    "identity_name", "identity", "name", "person", "person_name", "identity name",
]


def normalize_name(name) -> str:
    """Same normalization style as your original clean_name() rename step."""
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


def find_name_column(df, filepath):
    lower_map = {c.strip().lower(): c for c in df.columns}
    for cand in NAME_COLUMN_CANDIDATES:
        if cand in lower_map:
            return lower_map[cand]
    sys.exit(
        f"ERROR: could not find an identity-name column in {filepath}\n"
        f"Columns found: {list(df.columns)}\n"
        f"Expected one of: {NAME_COLUMN_CANDIDATES}\n"
        f"Either rename the column in the file, or tell me the real name."
    )


def load_pool(path: Path) -> set:
    if not path.exists():
        sys.exit(f"ERROR: file not found: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        sys.exit(f"ERROR: unsupported file type: {path}")
    col = find_name_column(df, path)
    normalized = df[col].apply(normalize_name)
    normalized = {n for n in normalized if n}
    return normalized


def check_pair(name_a, ids_a, name_b, ids_b, required, lines):
    overlap = ids_a & ids_b
    status = "PASS - disjoint" if not overlap else f"FAIL - {len(overlap)} shared identities"
    tag = "[REQUIRED]" if required else "[info]"
    line = f"{name_a:<14} vs {name_b:<14}: {status:<28} {tag}"
    print(line)
    lines.append(line)
    if overlap:
        sample = sorted(overlap)[:10]
        detail = f"    e.g. {sample}"
        print(detail)
        lines.append(detail)
    return bool(overlap) if required else False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=Path, help="Folder containing the metadata files")
    ap.add_argument("--report", default="leakage_report.txt", type=Path)
    args = ap.parse_args()

    print("Loading and normalizing identity pools...\n")
    pools = {}
    for pool_name, filename in FILES.items():
        path = args.dir / filename
        ids = load_pool(path)
        pools[pool_name] = ids
        print(f"  {pool_name:<14}: {len(ids):>6,} unique identities  ({filename})")

    # Combined DFFD_A pool (all its own splits) -- used when checking against
    # external datasets (DFF_C, DiffFace), since any DFFD_A identity anywhere
    # is off-limits for something ELSE's train/val set.
    dffd_a_all = pools["dffd_a_train"] | pools["dffd_a_val"] | pools["dffd_a_test"]
    dffd_b_all = pools["dffd_b"]

    print("\n" + "=" * 70)
    print("REQUIRED CHECKS  (these MUST show PASS)")
    print("=" * 70)
    lines = []
    any_fail = False

    # DFF_C trained -> evaluated on DiffFace
    any_fail |= check_pair("dff_c_train", pools["dff_c_train"], "diffface", pools["diffface"], True, lines)
    any_fail |= check_pair("dff_c_val",   pools["dff_c_val"],   "diffface", pools["diffface"], True, lines)

    # DFF_C trained -> evaluated on DFFD_A and DFFD_B
    any_fail |= check_pair("dff_c_train", pools["dff_c_train"], "dffd_a(all)", dffd_a_all, True, lines)
    any_fail |= check_pair("dff_c_val",   pools["dff_c_val"],   "dffd_a(all)", dffd_a_all, True, lines)
    any_fail |= check_pair("dff_c_train", pools["dff_c_train"], "dffd_b", dffd_b_all, True, lines)
    any_fail |= check_pair("dff_c_val",   pools["dff_c_val"],   "dffd_b", dffd_b_all, True, lines)

    # DFFD_A trained -> evaluated on DFFD_B
    any_fail |= check_pair("dffd_a_train", pools["dffd_a_train"], "dffd_b", dffd_b_all, True, lines)
    any_fail |= check_pair("dffd_a_val",   pools["dffd_a_val"],   "dffd_b", dffd_b_all, True, lines)

    print("\n" + "=" * 70)
    print("INFORMATIONAL ONLY  (not required to pass, shown for transparency)")
    print("=" * 70)
    check_pair("dff_c_test", pools["dff_c_test"], "diffface", pools["diffface"], False, lines)
    check_pair("dff_c_test", pools["dff_c_test"], "dffd_a(all)", dffd_a_all, False, lines)
    check_pair("dff_c_test", pools["dff_c_test"], "dffd_b", dffd_b_all, False, lines)
    check_pair("dffd_a_test", pools["dffd_a_test"], "dffd_b", dffd_b_all, False, lines)
    check_pair("diffface", pools["diffface"], "dffd_a(all)", dffd_a_all, False, lines)
    check_pair("diffface", pools["diffface"], "dffd_b", dffd_b_all, False, lines)

    print("\n" + "=" * 70)
    if any_fail:
        print("FINAL VERDICT: LEAKAGE DETECTED -- fix before responding to the reviewer.")
    else:
        print("FINAL VERDICT: NO LEAKAGE -- safe to state identity-disjoint protocol was verified.")
    print("=" * 70)

    with open(args.report, "w", encoding="utf-8") as f:
        f.write("Full cross-dataset identity leakage report\n" + "=" * 44 + "\n")
        f.write("\n".join(lines))
        f.write("\n\nFinal verdict: ")
        f.write("LEAKAGE DETECTED\n" if any_fail else "NO LEAKAGE\n")
    print(f"\nFull report saved to: {args.report}")

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
