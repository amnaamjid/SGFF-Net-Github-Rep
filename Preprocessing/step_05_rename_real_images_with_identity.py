import os
import re
import shutil
import unicodedata
import pandas as pd

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = "/home/amna/Dataset/DFFD/DFFD/DFFD_A_1/Original"

REAL_DIR = os.path.join(BASE_DIR, "Real")

IMDB_CSV = os.path.join(BASE_DIR, "imdb_metadata.csv")
WIKI_CSV = os.path.join(BASE_DIR, "wiki_metadata.csv")


# ==========================================================
# Convert person name into filename-safe format
# ==========================================================

def clean_name(name):
    """
    Convert person's name into a safe filename.

    Example:
        "Fred Astaire" -> "Fred_Astaire"
        "Sami Jauhojärvi" -> "Sami_Jauhojarvi"
    """

    if pd.isna(name):
        return "Unknown"

    name = str(name)

    # Remove accents
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")

    # Replace spaces with underscores
    name = name.replace(" ", "_")

    # Remove illegal filename characters
    name = re.sub(r'[<>:"/\\|?*]', "", name)

    # Remove duplicate underscores
    name = re.sub(r"_+", "_", name)

    # Remove leading/trailing underscores
    name = name.strip("_")

    return name


# ==========================================================
# Read metadata
# ==========================================================

def build_lookup(csv_path):

    df = pd.read_csv(csv_path)

    lookup = {}

    for _, row in df.iterrows():

        image_path = str(row["image_path"])

        filename = os.path.basename(image_path)

        person_name = clean_name(row["name"])

        lookup[filename] = person_name

    return lookup


print("Reading metadata...")

lookup = {}

lookup.update(build_lookup(IMDB_CSV))
lookup.update(build_lookup(WIKI_CSV))

print(f"Total metadata entries : {len(lookup):,}")


# ==========================================================
# Rename files
# ==========================================================

renamed = 0
already = 0
missing = 0

print("\nRenaming images...\n")

for filename in sorted(os.listdir(REAL_DIR)):

    old_path = os.path.join(REAL_DIR, filename)

    if not os.path.isfile(old_path):
        continue

    base, ext = os.path.splitext(filename)

    # Skip already renamed files
    if filename not in lookup:

        found = False

        for original_name, person in lookup.items():

            original_base = os.path.splitext(original_name)[0]

            if base.startswith(original_base + "_"):

                already += 1
                found = True
                break

        if found:
            continue

        missing += 1
        print(f"[Missing] {filename}")
        continue

    person = lookup[filename]

    new_filename = f"{base}_{person}{ext}"

    new_path = os.path.join(REAL_DIR, new_filename)

    if os.path.exists(new_path):
        already += 1
        continue

    shutil.move(old_path, new_path)

    renamed += 1

    if renamed % 500 == 0:
        print(f"Renamed {renamed:,} images...")


# ==========================================================
# Summary
# ==========================================================

print("\n=====================================")
print("Finished")
print("=====================================")
print(f"Renamed          : {renamed:,}")
print(f"Already renamed  : {already:,}")
print(f"Missing metadata : {missing:,}")
print("=====================================")
