import os
import shutil
import pandas as pd


# ===========================
# Paths
# ===========================

CSV_FILE = "master_identity_filtered.csv"

SOURCE_DIR = "Celeb-HQ-Real"

DEST_A = "Celeb-HQ-Real/DFFD_A"
DEST_B = "Celeb-HQ-Real/DFFD_B"


# Create folders
os.makedirs(DEST_A, exist_ok=True)
os.makedirs(DEST_B, exist_ok=True)


# ===========================
# Load CSV
# ===========================

print("Loading master identity file...")

df = pd.read_csv(CSV_FILE)

print(f"Rows: {len(df):,}")


# ===========================
# Get unique identities
# ===========================

identities = df["identity"].drop_duplicates().tolist()

print(f"Unique identities: {len(identities):,}")


# ===========================
# Identity-level split
# ===========================

identities_A = set(identities[:3000])
identities_B = set(identities[3000:])


print(f"DFFD_A identities: {len(identities_A):,}")
print(f"DFFD_B identities: {len(identities_B):,}")


# ===========================
# Copy images
# ===========================

count_A = 0
count_B = 0
missing = 0


print("\nCopying images...")


for _, row in df.iterrows():

    identity = row["identity"]

    # Convert:
    # 0.jpg + Olivia_Culpo
    # into:
    # 0_Olivia_Culpo.jpg

    image_id = os.path.splitext(row["real_image"])[0]

    filename = f"{image_id}_{identity}.jpg"


    source_path = os.path.join(
        SOURCE_DIR,
        filename
    )


    if not os.path.exists(source_path):

        print(f"Missing: {filename}")
        missing += 1
        continue


    # DFFD_A

    if identity in identities_A:

        shutil.copy2(
            source_path,
            os.path.join(DEST_A, filename)
        )

        count_A += 1


    # DFFD_B

    elif identity in identities_B:

        shutil.copy2(
            source_path,
            os.path.join(DEST_B, filename)
        )

        count_B += 1



# ===========================
# Results
# ===========================

print("\n==========================")
print("Finished")
print("==========================")

print(f"Images in DFFD_A : {count_A:,}")
print(f"Images in DFFD_B : {count_B:,}")
print(f"Total copied     : {count_A + count_B:,}")
print(f"Missing images   : {missing:,}")