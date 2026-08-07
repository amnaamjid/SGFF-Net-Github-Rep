import os
import shutil


# ===========================
# Paths
# ===========================

DFFD_A = "DFFD_A"

PGGAN_V1 = "pggan_v1"
PGGAN_V2 = "pggan_v2"

OUT_V1 = "pggan_v1_selected"
OUT_V2 = "pggan_v2_selected"


os.makedirs(OUT_V1, exist_ok=True)
os.makedirs(OUT_V2, exist_ok=True)



# ===========================
# Extract DFFD_A identities
# ===========================

print("Reading DFFD_A identities...")


real_identities = set()


for img in os.listdir(DFFD_A):

    if img.endswith(".jpg"):

        name = os.path.splitext(img)[0]

        # remove numeric index
        identity = "_".join(name.split("_")[1:])

        real_identities.add(identity)


print(f"DFFD_A identities found: {len(real_identities)}")



# ===========================
# Extract PGAN identity
# ===========================

def get_pgan_identity(filename):

    name = os.path.splitext(filename)[0]


    # remove PGAN part
    # Example:
    # Aaron_Carter_F_PGN1_03262
    # Aaron_Carter

    if "_F_PGN" in name:

        identity = name.split("_F_PGN")[0]

    else:
        return None


    return identity



# ===========================
# Process PGAN
# ===========================

def process_pgan(source, output, version):

    copied = 0
    matched_ids = set()


    print("\nProcessing", version)


    for img in os.listdir(source):

        if not img.endswith(".jpg"):
            continue


        pgan_identity = get_pgan_identity(img)


        if pgan_identity is None:
            continue


        matched = None


        # direct match
        if pgan_identity in real_identities:

            matched = pgan_identity


        # remove first token:
        # A_Jerrold_Perenchio
        # becomes
        # Jerrold_Perenchio

        elif "_" in pgan_identity:

            without_prefix = "_".join(
                pgan_identity.split("_")[1:]
            )

            if without_prefix in real_identities:

                matched = without_prefix



        if matched:


            folder = os.path.join(
                output,
                matched
            )

            os.makedirs(
                folder,
                exist_ok=True
            )


            shutil.copy2(
                os.path.join(source,img),
                os.path.join(folder,img)
            )


            copied += 1
            matched_ids.add(matched)



    print("Copied images:", copied)
    print("Matched identities:", len(matched_ids))



# ===========================
# Run
# ===========================

process_pgan(
    PGGAN_V1,
    OUT_V1,
    "PGGAN_v1"
)


process_pgan(
    PGGAN_V2,
    OUT_V2,
    "PGGAN_v2"
)


print("\nFinished")