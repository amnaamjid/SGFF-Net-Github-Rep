"""
majority_vote.py
Combines two or more annotators' CSV files (from results/) into a single
final_annotations.csv, using majority voting on the "Selected" column
(the annotator's final best-<=3 choice per identity/type).

Usage:
    python majority_vote.py
    python majority_vote.py --results-dir results --output final_annotations.csv
    python majority_vote.py --files results/Annotator1.csv results/Annotator2.csv results/Annotator3.csv

Rules:
- An image's final status is KEEP if a MAJORITY of annotators marked it
  Selected=Yes (i.e. more than half of the annotators who annotated it).
- Ties (e.g. 1 of 2 annotators, or 2 annotators disagreeing with no 3rd
  vote available) are reported as "TIE" and left for manual review -
  they are NOT silently discarded or kept.
- If, after majority voting, MORE than MAX_KEEP_PER_TYPE images per
  identity/type are marked KEEP, the ones with the highest vote count
  (then highest average score) are kept, up to the limit, and the rest
  are downgraded to DISCARD with a note in the Reason column.
"""

import os
import glob
import argparse
import pandas as pd
import config


def find_result_files(results_dir):
    return sorted(glob.glob(os.path.join(results_dir, "*.csv")))


def load_all(files):
    frames = []
    for f in files:
        df = pd.read_csv(f, dtype=str).fillna("")
        df["__annotator"] = os.path.splitext(os.path.basename(f))[0]
        frames.append(df)
    if not frames:
        raise SystemExit("No annotator CSV files found.")
    return pd.concat(frames, ignore_index=True)


def _row_meets_criteria(row):
    """Recomputes whether this single annotator's raw answers meet the
    KEEP criteria for this image type - the CSVs no longer store a
    precomputed Score/Decision, so this is derived fresh from the raw
    Yes/No answers, using the exact same rule the app used live."""
    answers = {
        "Identity": row.get("Identity", ""),
        "Reliable": row.get("Reliable", ""),
        "SingleFace": row.get("SingleFace", ""),
        "Quality": row.get("Quality", ""),
        "Artifact": row.get("Artifact", ""),
    }
    return config.meets_criteria(row.get("Type", ""), answers)


def majority_vote(all_df):
    records = []
    group_cols = ["ID", "Celebrity", "Image", "Type"]

    for (identity_id, celebrity, image, image_type), group in all_df.groupby(group_cols):
        n = len(group)
        n_yes = (group["Selected"] == "Yes").sum()
        n_no = (group["Selected"] == "No").sum()
        criteria_met = sum(1 for _, r in group.iterrows() if _row_meets_criteria(r))
        criteria_met_fraction = round(criteria_met / n, 2) if n else 0

        if n_yes > n / 2:
            final_decision = "KEEP"
        elif n_no >= n / 2 and n_yes <= n / 2 and n_yes != n_no:
            final_decision = "DISCARD"
        elif n_yes == n_no:
            final_decision = "TIE"
        else:
            final_decision = "DISCARD"

        records.append({
            "ID": identity_id,
            "Celebrity": celebrity,
            "Image": image,
            "Type": image_type,
            "NumAnnotators": n,
            "VotesYes": n_yes,
            "VotesNo": n_no,
            "CriteriaMetFraction": criteria_met_fraction,
            "FinalDecision": final_decision,
            "Reason": "",
        })

    return pd.DataFrame(records)


def enforce_max_per_type(final_df, max_keep=config.MAX_KEEP_PER_TYPE):
    """If majority voting leaves more than max_keep KEEPs for an identity/type,
    keep only the top-voted (then highest criteria-met-fraction) ones."""
    final_df = final_df.copy()
    for (identity_id, image_type), group in final_df.groupby(["ID", "Type"]):
        kept = group[group["FinalDecision"] == "KEEP"]
        if len(kept) <= max_keep:
            continue
        ranked = kept.sort_values(
            by=["VotesYes", "CriteriaMetFraction"], ascending=[False, False]
        )
        overflow_idx = ranked.index[max_keep:]
        final_df.loc[overflow_idx, "FinalDecision"] = "DISCARD"
        final_df.loc[overflow_idx, "Reason"] = f"Exceeded max {max_keep} per identity/type after voting"
    return final_df


def main():
    parser = argparse.ArgumentParser(description="Majority-vote annotator CSVs into a final dataset list.")
    parser.add_argument("--results-dir", default=config.RESULTS_DIR,
                         help="Folder containing annotator CSV files (default: results/)")
    parser.add_argument("--files", nargs="+", default=None,
                         help="Explicit list of annotator CSV files (overrides --results-dir)")
    parser.add_argument("--output", default=os.path.join(config.BASE_DIR, "final_annotations.csv"),
                         help="Output CSV path")
    args = parser.parse_args()

    files = args.files if args.files else find_result_files(args.results_dir)
    if not files:
        raise SystemExit(f"No CSV files found in {args.results_dir}")

    print(f"Loading {len(files)} annotator file(s):")
    for f in files:
        print(f"  - {f}")

    all_df = load_all(files)
    final_df = majority_vote(all_df)
    final_df = enforce_max_per_type(final_df)

    final_df.to_csv(args.output, index=False)

    n_keep = (final_df["FinalDecision"] == "KEEP").sum()
    n_discard = (final_df["FinalDecision"] == "DISCARD").sum()
    n_tie = (final_df["FinalDecision"] == "TIE").sum()

    print("\nDone.")
    print(f"  KEEP:    {n_keep}")
    print(f"  DISCARD: {n_discard}")
    print(f"  TIE (needs manual review): {n_tie}")
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
