#!/usr/bin/env python3
"""
build_diffface.py

Builds DiffFace_A .. DiffFace_F datasets from:
    <ROOT>/Celeb-HQ-Real/                (24,721 real images)
    <ROOT>/Fake/ADM
    <ROOT>/Fake/DDIM
    <ROOT>/Fake/DDPM
    <ROOT>/Fake/LDM
    <ROOT>/Fake/PNDM

Logic:
    1. Find the set of real images whose filename (stem, extension-agnostic)
       has a matching fake image in ALL FIVE generator folders
       (ADM, DDIM, DDPM, LDM, PNDM).
    2. Randomly select N_SELECT (default 3000) of those real images
       (fixed random seed => reproducible).
    3. Create:
         DiffFace_A/real, DiffFace_A/fake   <- fake = randomly shuffled
                                                mix across all 5 generators
                                                (one generator per image)
         DiffFace_B/real, DiffFace_B/fake   <- fake always from ADM
         DiffFace_C/real, DiffFace_C/fake   <- fake always from DDPM
         DiffFace_D/real, DiffFace_D/fake   <- fake always from LDM
         DiffFace_E/real, DiffFace_E/fake   <- fake always from PNDM
         DiffFace_F/real, DiffFace_F/fake   <- fake always from DDIM
       All six "real" folders contain the EXACT SAME 3000 real images.
    4. Moves (not copies) the 3000 selected real images out of
       Celeb-HQ-Real into Celeb-HQ-Real-Selected, so they no longer sit
       in the original pool.
    5. Writes a text file listing exactly which real filenames were chosen,
       and which generator was assigned to each one in DiffFace_A.

SAFE BY DEFAULT: DRY_RUN = True below. Run once in dry-run to sanity check
counts, then flip to False to actually copy/move files.
"""

import os
import random
import shutil
from pathlib import Path

# ------------------------------------------------------------------ #
# CONFIG - edit these paths / settings
# ------------------------------------------------------------------ #
ROOT = Path.home() / "Dataset" / "DFFD-Diffface"

REAL_DIR = ROOT / "Celeb-HQ-Real"
FAKE_DIR = ROOT / "Fake"

# generator folders that MUST all contain a match for a real image to qualify
GENERATORS = ["ADM", "DDIM", "DDPM", "LDM", "PNDM"]

# which generator feeds which single-generator DiffFace folder
SINGLE_GEN_MAP = {
    "DiffFace_B": "ADM",
    "DiffFace_C": "DDPM",
    "DiffFace_D": "LDM",
    "DiffFace_E": "PNDM",
    "DiffFace_F": "DDIM",
}

N_SELECT = 3000
RANDOM_SEED = 42

OUTPUT_ROOT = ROOT  # DiffFace_A..F will be created directly under here
MOVED_REAL_DIR = ROOT / "Celeb-HQ-Real-Selected"  # where selected reals get moved to

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DRY_RUN = False  # <-- set to False once counts look right, to actually run it
# ------------------------------------------------------------------ #


def list_images_by_stem(folder: Path) -> dict:
    """Return {filename_stem: full_path} for all image files in folder."""
    out = {}
    if not folder.exists():
        print(f"  [WARN] folder does not exist: {folder}")
        return out
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in VALID_EXTS:
            out[f.stem] = f
    return out


def main():
    random.seed(RANDOM_SEED)

    print("=" * 70)
    print("STEP 1: Indexing real images and fake images per generator")
    print("=" * 70)

    real_index = list_images_by_stem(REAL_DIR)
    print(f"Real images found in Celeb-HQ-Real: {len(real_index)}")

    gen_index = {}
    for gen in GENERATORS:
        idx = list_images_by_stem(FAKE_DIR / gen)
        gen_index[gen] = idx
        print(f"Fake images found in {gen}: {len(idx)}")

    print()
    print("=" * 70)
    print("STEP 2: Finding real images that have a match in ALL 5 generators")
    print("=" * 70)

    # intersection of stems present in real AND in every generator folder
    common_stems = set(real_index.keys())
    for gen in GENERATORS:
        common_stems &= set(gen_index[gen].keys())

    common_stems = sorted(common_stems)  # sorted for reproducibility before shuffle
    print(f"Real images with a match in ALL of {GENERATORS}: {len(common_stems)}")

    if len(common_stems) < N_SELECT:
        raise SystemExit(
            f"ERROR: only {len(common_stems)} real images have matches in all "
            f"5 generators, but you asked for {N_SELECT}. "
            f"Lower N_SELECT or check filename matching."
        )

    print()
    print("=" * 70)
    print(f"STEP 3: Randomly selecting {N_SELECT} real images (seed={RANDOM_SEED})")
    print("=" * 70)

    selected_stems = random.sample(common_stems, N_SELECT)
    selected_stems_set = set(selected_stems)
    print(f"Selected {len(selected_stems)} real images.")

    # Assign a random generator to each selected stem, for DiffFace_A's mixed fakes
    a_assignment = {stem: random.choice(GENERATORS) for stem in selected_stems}

    print()
    print("=" * 70)
    print("STEP 4: Planned folder structure")
    print("=" * 70)
    diffface_folders = ["DiffFace_A", "DiffFace_B", "DiffFace_C",
                         "DiffFace_D", "DiffFace_E", "DiffFace_F"]
    for name in diffface_folders:
        print(f"  {OUTPUT_ROOT / name / 'real'}  ({N_SELECT} images)")
        print(f"  {OUTPUT_ROOT / name / 'fake'}  ({N_SELECT} images)")
    print(f"  {MOVED_REAL_DIR}  <- selected real images moved here from Celeb-HQ-Real")

    if DRY_RUN:
        print()
        print("*** DRY_RUN = True : no files copied or moved. ***")
        print("*** Review the counts above. Set DRY_RUN = False to actually run. ***")
        write_selection_log(selected_stems, a_assignment, real_index, dry_run=True)
        return

    print()
    print("=" * 70)
    print("STEP 5: Creating folders and copying files")
    print("=" * 70)

    for name in diffface_folders:
        (OUTPUT_ROOT / name / "real").mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / name / "fake").mkdir(parents=True, exist_ok=True)
    MOVED_REAL_DIR.mkdir(parents=True, exist_ok=True)

    for i, stem in enumerate(selected_stems, 1):
        real_src = real_index[stem]

        # copy the real image into every DiffFace_X/real
        for name in diffface_folders:
            dst = OUTPUT_ROOT / name / "real" / real_src.name
            shutil.copy2(real_src, dst)

        # DiffFace_A fake: mixed generator
        gen_for_a = a_assignment[stem]
        fake_src_a = gen_index[gen_for_a][stem]
        shutil.copy2(fake_src_a, OUTPUT_ROOT / "DiffFace_A" / "fake" / fake_src_a.name)

        # DiffFace_B..F fake: fixed single generator
        for folder_name, gen_name in SINGLE_GEN_MAP.items():
            fake_src = gen_index[gen_name][stem]
            shutil.copy2(fake_src, OUTPUT_ROOT / folder_name / "fake" / fake_src.name)

        if i % 250 == 0 or i == len(selected_stems):
            print(f"  processed {i}/{len(selected_stems)}")

    print()
    print("=" * 70)
    print("STEP 6: Moving selected real images out of Celeb-HQ-Real")
    print("=" * 70)
    for stem in selected_stems:
        real_src = real_index[stem]
        dst = MOVED_REAL_DIR / real_src.name
        shutil.move(str(real_src), str(dst))
    print(f"Moved {len(selected_stems)} real images to {MOVED_REAL_DIR}")

    write_selection_log(selected_stems, a_assignment, real_index, dry_run=False)

    print()
    print("DONE.")


def write_selection_log(selected_stems, a_assignment, real_index, dry_run):
    log_path = ROOT / "diffface_selected_real_images.txt"
    lines = []
    lines.append(f"# Selected {len(selected_stems)} real images (seed={RANDOM_SEED})")
    lines.append(f"# dry_run={dry_run}")
    lines.append("# stem, original_filename, generator_used_in_DiffFace_A")
    for stem in selected_stems:
        fname = real_index[stem].name if stem in real_index else stem
        lines.append(f"{stem}, {fname}, {a_assignment[stem]}")
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Selection log written to: {log_path}")


if __name__ == "__main__":
    main()
