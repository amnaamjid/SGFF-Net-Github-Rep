#!/usr/bin/env python3

import os
import pandas as pd

# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = "/home/amna/HPC_DATA/DFF"

REAL_DIR = os.path.join(BASE_DIR, "Real")

IMDB_CSV = os.path.join(BASE_DIR, "imdb_metadata.csv")
WIKI_CSV = os.path.join(BASE_DIR, "wiki_metadata.csv")

OUTPUT_CSV = os.path.join(BASE_DIR, "real_identity_metadata.csv")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ==========================================================
# LOAD METADATA
# ==========================================================

def load_metadata(csv_file, source):

    print(f"Loading {source} metadata...")

    df = pd.read_csv(csv_file)

    df["source"] = source
    df["original_filename"] = df["image_path"].apply(os.path.basename)

    print(f"  {len(df):,} rows loaded")

    return df


imdb = load_metadata(IMDB_CSV, "IMDB")
wiki = load_metadata(WIKI_CSV, "Wiki")

metadata = pd.concat([imdb, wiki], ignore_index=True)

print(f"\nTotal metadata rows : {len(metadata):,}")

# ==========================================================
# BUILD LOOKUP
# ==========================================================

print("Building lookup dictionary...")

lookup = {}

duplicate_metadata = 0

for row in metadata.itertuples(index=False):

    filename = row.original_filename

    if filename in lookup:
        duplicate_metadata += 1
        continue

    lookup[filename] = {
        "identity": row.name,
        "source": row.source,
        "gender": row.gender,
        "photo_taken": row.photo_taken,
        "dob": row.dob,
        "face_score": row.face_score,
        "second_face_score": row.second_face_score,
    }

print(f"Lookup entries      : {len(lookup):,}")
print(f"Duplicate metadata  : {duplicate_metadata:,}")

# ==========================================================
# PROCESS REAL IMAGES
# ==========================================================

print("\nScanning Real folder...")

image_files = sorted([
    f for f in os.listdir(REAL_DIR)
    if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
])

print(f"Images found : {len(image_files):,}\n")

rows = []

matched = 0
unknown = 0

for i, filename in enumerate(image_files, start=1):

    # Remove appended identity if present.
    # Example:
    # 10002116_1971-05-31_2012_Diana_Damrau.jpg
    # ->
    # 10002116_1971-05-31_2012.jpg

    original_filename = filename

    while original_filename not in lookup:

        base, ext = os.path.splitext(original_filename)

        if "_" not in base:
            break

        base = base.rsplit("_", 1)[0]
        original_filename = base + ext

    if original_filename not in lookup:
        unknown += 1
        continue

    info = lookup[original_filename]

    rows.append({
        "filename": filename,
        "original_filename": original_filename,
        "identity": info["identity"],
        "source": info["source"],
        "gender": info["gender"],
        "photo_taken": info["photo_taken"],
        "dob": info["dob"],
        "face_score": info["face_score"],
        "second_face_score": info["second_face_score"]
    })

    matched += 1

    if i % 5000 == 0:
        print(f"Processed {i:,}/{len(image_files):,}")

# ==========================================================
# SAVE
# ==========================================================

print("\nSaving CSV...")

df = pd.DataFrame(rows)

df.sort_values(
    ["identity", "filename"],
    inplace=True,
    ignore_index=True
)

df.to_csv(OUTPUT_CSV, index=False)

# ==========================================================
# STATISTICS
# ==========================================================

print("\n" + "=" * 60)

print("SUMMARY")

print("=" * 60)

print(f"Images in Real folder        : {len(image_files):,}")
print(f"Matched images              : {matched:,}")
print(f"Unknown images              : {unknown:,}")

print()

print(f"Unique identities           : {df['identity'].nunique():,}")

print(f"IMDB images                 : {(df.source=='IMDB').sum():,}")
print(f"Wiki images                 : {(df.source=='Wiki').sum():,}")

print(f"IMDB identities             : {df[df.source=='IMDB']['identity'].nunique():,}")
print(f"Wiki identities             : {df[df.source=='Wiki']['identity'].nunique():,}")

print()

print(f"CSV saved to:")
print(OUTPUT_CSV)

print("=" * 60)