import os
import random
import pandas as pd

# ==========================================================
# SETTINGS
# ==========================================================

FOLDER = os.getcwd()                      # Current folder
OUTPUT = os.path.join(FOLDER, "Master_Sheet_Unique.xlsx")

TARGET_SIZE = 500

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

print("=" * 70)
print("Working Folder :", FOLDER)
print("Output File    :", OUTPUT)
print("=" * 70)

# ==========================================================
# FIND ALL EXCEL FILES
# ==========================================================

excel_files = sorted([
    f for f in os.listdir(FOLDER)
    if f.endswith(".xlsx") or f.endswith(".xls")
])

# Ignore output file if already exists
excel_files = [
    f for f in excel_files
    if f.lower() != "master_sheet.xlsx"
]

print(f"\nFound {len(excel_files)} Excel files.\n")

# ==========================================================
# READ ALL FILES
# ==========================================================

sheet_data = []
total_rows = 0

for file in excel_files:

    path = os.path.join(FOLDER, file)

    print(f"Reading: {file}")

    try:
        df = pd.read_excel(path)
    except Exception as e:
        print(f"Cannot read {file}")
        print(e)
        continue

    # Check required columns
    if "Name" not in df.columns:
        print(f"Skipping {file} (No Name column)")
        continue

    # Create missing columns if needed
    if "Profession" not in df.columns:
        df["Profession"] = ""

    if "Gender" not in df.columns:
        df["Gender"] = ""

    # Keep only required columns
    df = df[["Name", "Profession", "Gender"]].copy()

    # Remove empty names
    df["Name"] = df["Name"].fillna("").astype(str).str.strip()

    df = df[df["Name"] != ""]

    # Fill empty profession/gender with blank
    df["Profession"] = df["Profession"].fillna("")
    df["Gender"] = df["Gender"].fillna("")

    # Remove duplicates INSIDE each sheet (case insensitive)
    df["key"] = df["Name"].str.lower().str.strip()

    df = df.drop_duplicates(subset="key")

    df = df.reset_index(drop=True)

    sheet_data.append({
        "file": file,
        "data": df,
        "count": len(df)
    })

    total_rows += len(df)

print("\n==========================================")
print("Total Available Unique Names:", total_rows)
print("==========================================")

# ==========================================================
# PROPORTIONAL RANDOM SAMPLING
# ==========================================================

selected_rows = []

print("\nSampling...\n")

for sheet in sheet_data:

    proportion = sheet["count"] / total_rows

    sample_size = round(proportion * TARGET_SIZE)

    sample_size = min(sample_size, sheet["count"])

    sampled = sheet["data"].sample(
        n=sample_size,
        random_state=RANDOM_SEED
    )

    selected_rows.append(sampled)

    print(f"{sheet['file']}")
    print(f"   Total Rows : {sheet['count']}")
    print(f"   Sample Size: {sample_size}\n")

selected_df = pd.concat(selected_rows, ignore_index=True)

print("Initially Sampled:", len(selected_df))

# ==========================================================
# REMOVE DUPLICATES ACROSS ALL SHEETS
# ==========================================================

selected_df["key"] = selected_df["Name"].str.lower().str.strip()

selected_df = selected_df.drop_duplicates(subset="key")

print("After Duplicate Removal:", len(selected_df))

# ==========================================================
# FILL REMAINING NAMES IF NEEDED
# ==========================================================

if len(selected_df) < TARGET_SIZE:

    print("\nAdding remaining unique names...")

    all_data = []

    for sheet in sheet_data:
        all_data.append(sheet["data"])

    all_df = pd.concat(all_data, ignore_index=True)

    all_df["key"] = all_df["Name"].str.lower().str.strip()

    all_df = all_df.drop_duplicates(subset="key")

    remaining = all_df[
        ~all_df["key"].isin(selected_df["key"])
    ]

    remaining = remaining.sample(
        frac=1,
        random_state=RANDOM_SEED
    )

    needed = TARGET_SIZE - len(selected_df)

    selected_df = pd.concat(
        [selected_df, remaining.head(needed)],
        ignore_index=True
    )

# ==========================================================
# FINAL SHUFFLE
# ==========================================================

selected_df = selected_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

# Keep only required number
selected_df = selected_df.head(TARGET_SIZE)

# ==========================================================
# CREATE MASTER SHEET
# ==========================================================

master = selected_df[["Name", "Profession", "Gender"]].copy()

master.insert(0, "S.No", range(1, len(master) + 1))

master.to_excel(OUTPUT, index=False)

print("\n==========================================")
print("Master Sheet Created Successfully!")
print("Total Selected :", len(master))
print("Saved To       :", OUTPUT)
print("==========================================")