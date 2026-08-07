import pandas as pd

# File path
csv_file = "common_identities_between_dffc_and_celeba.csv"

# Read CSV
df = pd.read_csv(csv_file)

print("=" * 50)
print("Dataset Statistics")
print("=" * 50)

# Total rows
print(f"Total rows            : {len(df)}")

# Unique identities
unique_ids = df["identity"].nunique()
print(f"Unique identities     : {unique_ids}")

# Duplicate rows (same identity appearing multiple times)
duplicate_rows = len(df) - unique_ids
print(f"Duplicate entries     : {duplicate_rows}")

print("\nTop 20 identities with most images:")
print(df["identity"].value_counts().head(20))

print("\nIdentity frequency summary")
print(df["identity"].value_counts().describe())

# Save frequency table
freq = (
    df["identity"]
    .value_counts()
    .reset_index()
)
freq.columns = ["identity", "num_images"]

freq.to_csv("identity_frequency.csv", index=False)

print("\nSaved identity frequencies to:")
print("identity_frequency.csv")
