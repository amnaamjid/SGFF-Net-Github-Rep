import os
import shutil


# ===========================
# Paths
# ===========================

DFFD_B = "DFFD_B"

STYLEGAN = "stylegan"
STARGAN = "stargan"

OUT_STYLEGAN = "stylegan_selected"
OUT_STARGAN = "stargan_selected"


os.makedirs(OUT_STYLEGAN, exist_ok=True)
os.makedirs(OUT_STARGAN, exist_ok=True)



# ===========================
# Read real identities
# ===========================

print("Reading DFFD_B identities...")


real_identity_map = {}


for img in os.listdir(DFFD_B):

    if not img.lower().endswith(".jpg"):
        continue


    name = os.path.splitext(img)[0]


    # Example:
    # 12747_Wanda_Hawley.jpg
    #
    # output:
    # Wanda_Hawley

    identity = "_".join(
        name.split("_")[1:]
    )


    key = identity.lower().strip()


    real_identity_map[key] = identity



print(
    "DFFD_B identities found:",
    len(real_identity_map)
)




# ===========================
# Extract StyleGAN identity
# ===========================

def get_stylegan_identity(filename):

    name = os.path.splitext(filename)[0]


    if "_F_SyCA" not in name:
        return None


    identity = name.split("_F_SyCA")[0]


    return identity




# ===========================
# Extract StarGAN identity
# ===========================

def get_stargan_identity(filename):

    name = os.path.splitext(filename)[0]


    if "_F_STGN" not in name:
        return None


    identity = name.split("_F_STGN")[0]


    # remove invalid names
    if identity == "":
        return None


    return identity




# ===========================
# Process fake dataset
# ===========================

def process_fake(
        source,
        output,
        extractor,
        dataset_name):


    copied = 0
    matched_ids = set()


    print("\nProcessing", dataset_name)



    for img in os.listdir(source):


        if not img.lower().endswith(
            (".jpg",".png")
        ):
            continue



        identity = extractor(img)



        if identity is None:
            continue



        key = identity.lower().strip()



        if key in real_identity_map:


            # use original real identity name
            folder_name = real_identity_map[key]


            out_folder = os.path.join(
                output,
                folder_name
            )


            os.makedirs(
                out_folder,
                exist_ok=True
            )



            shutil.copy2(
                os.path.join(source,img),
                os.path.join(out_folder,img)
            )


            copied += 1
            matched_ids.add(folder_name)



    print(
        "Copied images:",
        copied
    )

    print(
        "Matched identities:",
        len(matched_ids)
    )




# ===========================
# Run
# ===========================


process_fake(
    STYLEGAN,
    OUT_STYLEGAN,
    get_stylegan_identity,
    "StyleGAN"
)



process_fake(
    STARGAN,
    OUT_STARGAN,
    get_stargan_identity,
    "StarGAN"
)



print("\nFinished")
