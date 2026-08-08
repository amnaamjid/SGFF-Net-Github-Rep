"""
build_final_dataset.py
Takes the output of majority_vote.py (final_annotations.csv) and actually
BUILDS the final, cleaned-up dataset folder - copying only the images that
were voted KEEP into a fresh folder, organized exactly like your original
dataset (ID_Name/Real/... and ID_Name/Fake/...).

IMPORTANT RULE: an identity is only included in the final dataset if it
ends up with AT LEAST ONE final Real image AND AT LEAST ONE final Fake
image after voting. An identity with only one side (e.g. Real images
kept but zero Fake images kept) is NOT usable for a real-vs-fake dataset,
so it is excluded entirely rather than copied half-finished. Every
excluded identity is listed in the report with the reason, so you can
follow up (send it back for re-annotation, resolve a TIE, etc.) instead
of it silently going missing.

Use --allow-partial if you want the old behavior instead (copy whatever
was KEEP, even if only one side exists for that identity).

HOW TO USE
----------
1. Run majority_vote.py first, so you have final_annotations.csv.
2. Make sure your original Dataset/ folder (the one with all the source
   images) is present, so this script can find and copy the actual files.
3. Run:
       python build_final_dataset.py
4. Your final images appear in: FinalDataset/
   A report of anything needing your attention (TIEs, identities missing
   Real or Fake images, identities excluded entirely) is saved to:
       FinalDataset_Report.csv

Custom paths are supported - see the --help output:
    python build_final_dataset.py --help
"""

import os
import sys
import shutil
import argparse
import pandas as pd
import config


def find_source_image(dataset_dir, identity_id, image_type, filename):
    """
    Looks for <dataset_dir>/<ID..._AnyName>/<Real|Fake>/<filename>.
    The identity folder name only needs to START with the ID - this way it
    still works even if celebrity names/spelling differ slightly between
    your dataset folders and what's recorded in the CSVs.
    """
    if not os.path.isdir(dataset_dir):
        return None
    for entry in os.listdir(dataset_dir):
        if entry == identity_id or entry.startswith(identity_id + "_"):
            candidate = os.path.join(dataset_dir, entry, image_type, filename)
            if os.path.isfile(candidate):
                return candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Build the final dataset folder from majority_vote.py's output."
    )
    parser.add_argument("--final-csv", default=os.path.join(config.BASE_DIR, "final_annotations.csv"),
                         help="Path to final_annotations.csv (output of majority_vote.py)")
    parser.add_argument("--dataset-dir", default=config.DATASET_DIR,
                         help="Path to the ORIGINAL dataset folder (with all source images)")
    parser.add_argument("--output-dir", default=os.path.join(config.BASE_DIR, "FinalDataset"),
                         help="Where to create the cleaned-up final dataset")
    parser.add_argument("--report", default=os.path.join(config.BASE_DIR, "FinalDataset_Report.csv"),
                         help="Where to save the follow-up report (TIEs, excluded identities, etc.)")
    parser.add_argument("--allow-partial", action="store_true",
                         help="Include identities even if they only have Real OR only Fake images "
                              "kept (instead of requiring both). Off by default.")
    args = parser.parse_args()

    if not os.path.exists(args.final_csv):
        print(f"ERROR: could not find '{args.final_csv}'.")
        print("Run majority_vote.py first to produce final_annotations.csv.")
        sys.exit(1)

    if not os.path.isdir(args.dataset_dir):
        print(f"ERROR: could not find the original dataset folder at '{args.dataset_dir}'.")
        print("This script needs the ORIGINAL images to copy from, not just the CSV.")
        sys.exit(1)

    df = pd.read_csv(args.final_csv, dtype=str).fillna("")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- First pass: for every identity, count how many KEEP images each
    # type (Real/Fake) ended up with, so we can decide inclusion BEFORE
    # copying anything. ---
    keep_counts = {}   # (id, celebrity) -> {"Real": n, "Fake": n}
    for (identity_id, celebrity), group in df.groupby(["ID", "Celebrity"]):
        counts = {"Real": 0, "Fake": 0}
        for image_type in ("Real", "Fake"):
            type_group = group[group["Type"] == image_type]
            counts[image_type] = int((type_group["FinalDecision"] == "KEEP").sum())
        keep_counts[(identity_id, celebrity)] = counts

    excluded_identities = set()
    if not args.allow_partial:
        for key, counts in keep_counts.items():
            has_real_col = not df[(df["ID"] == key[0]) & (df["Type"] == "Real")].empty
            has_fake_col = not df[(df["ID"] == key[0]) & (df["Type"] == "Fake")].empty
            # Only enforce "needs both" for identities that actually HAVE both
            # types in the source data. An identity that never had any Fake
            # images to begin with isn't "missing" anything - it just doesn't
            # apply here, so it's judged on Real alone (and vice versa).
            missing_real = has_real_col and counts["Real"] == 0
            missing_fake = has_fake_col and counts["Fake"] == 0
            if missing_real or missing_fake:
                excluded_identities.add(key)

    # --- Second pass: copy files, skipping excluded identities entirely ---
    copied = 0
    missing_source = []
    report_rows = []

    for _, row in df.iterrows():
        identity_id = row["ID"]
        celebrity = row["Celebrity"]
        image = row["Image"]
        image_type = row["Type"]
        decision = row["FinalDecision"]

        if decision != "KEEP":
            continue
        if (identity_id, celebrity) in excluded_identities:
            continue

        source_path = find_source_image(args.dataset_dir, identity_id, image_type, image)
        if source_path is None:
            missing_source.append((identity_id, celebrity, image_type, image))
            continue

        safe_name = f"{identity_id}_{celebrity}".replace(" ", "_")
        dest_dir = os.path.join(args.output_dir, safe_name, image_type)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(source_path, os.path.join(dest_dir, image))
        copied += 1

    # --- Build the follow-up report ---
    ties = df[df["FinalDecision"] == "TIE"]
    for _, row in ties.iterrows():
        report_rows.append({
            "ID": row["ID"], "Celebrity": row["Celebrity"], "Image": row["Image"],
            "Type": row["Type"], "Issue": "TIE - annotators disagreed, needs a manual look",
        })

    for (identity_id, celebrity) in excluded_identities:
        counts = keep_counts[(identity_id, celebrity)]
        reason_parts = []
        if counts["Real"] == 0:
            reason_parts.append("0 Real images kept")
        if counts["Fake"] == 0:
            reason_parts.append("0 Fake images kept")
        report_rows.append({
            "ID": identity_id, "Celebrity": celebrity, "Image": "", "Type": "",
            "Issue": f"EXCLUDED from FinalDataset - needs both Real and Fake, but has: "
                     f"{' and '.join(reason_parts)} after voting",
        })

    for identity_id, celebrity, image_type, image in missing_source:
        report_rows.append({
            "ID": identity_id, "Celebrity": celebrity, "Image": image, "Type": image_type,
            "Issue": "KEEP decision, but the source image file could not be found - not copied",
        })

    report_df = pd.DataFrame(report_rows, columns=["ID", "Celebrity", "Image", "Type", "Issue"])
    report_df.to_csv(args.report, index=False)

    total_identities = df.groupby(["ID", "Celebrity"]).ngroups
    included_identities = total_identities - len(excluded_identities)

    print(f"\nDone. Copied {copied} final image(s) for {included_identities} of "
          f"{total_identities} identities into: {args.output_dir}")
    if excluded_identities:
        print(f"EXCLUDED {len(excluded_identities)} identity(ies) - each needs both Real and "
              f"Fake images kept, but at least one side had zero. See the report for exactly "
              f"which ones and why (use --allow-partial to include them anyway).")
    if missing_source:
        print(f"WARNING: {len(missing_source)} KEEP image(s) could not be found in the dataset "
              f"folder and were skipped - see the report.")
    print(f"Follow-up report ({len(report_rows)} item(s) needing attention) saved to: {args.report}")


if __name__ == "__main__":
    main()
