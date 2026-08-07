import os
import shutil
import random


random.seed(42)


INPUT = "DFFD_A_dataset"

OUTPUT = "DFFD_A_dataset_balanced"



# ==========================
# Helper
# ==========================

def get_identity(filename):

    name = os.path.splitext(filename)[0]


    # Real:
    # 10021_Ryan_Braun.jpg
    #
    # Fake:
    # Ryan_Braun_F_PGN1_12345.jpg

    if name[0].isdigit():

        identity = "_".join(
            name.split("_")[1:]
        )

    else:

        identity = name.split("_F_PGN")[0]


    return identity



# ==========================
# Copy validation/test
# ==========================

def copy_folder(split):

    for cls in ["real","fake"]:

        src = os.path.join(
            INPUT,
            split,
            cls
        )

        dst = os.path.join(
            OUTPUT,
            split,
            cls
        )

        os.makedirs(dst, exist_ok=True)


        for img in os.listdir(src):

            shutil.copy2(
                os.path.join(src,img),
                os.path.join(dst,img)
            )



# ==========================
# Identity balancing train
# ==========================

def balance_train():


    real_dir = os.path.join(
        INPUT,
        "train",
        "real"
    )

    fake_dir = os.path.join(
        INPUT,
        "train",
        "fake"
    )


    out_real = os.path.join(
        OUTPUT,
        "train",
        "real"
    )

    out_fake = os.path.join(
        OUTPUT,
        "train",
        "fake"
    )


    os.makedirs(out_real, exist_ok=True)
    os.makedirs(out_fake, exist_ok=True)



    # --------------------
    # Group real images
    # --------------------

    real_ids = {}


    for img in os.listdir(real_dir):

        identity = get_identity(img)

        real_ids.setdefault(
            identity,
            []
        ).append(img)



    # --------------------
    # Group fake images
    # --------------------

    fake_ids = {}


    for img in os.listdir(fake_dir):

        identity = get_identity(img)

        fake_ids.setdefault(
            identity,
            []
        ).append(img)



    print(
        "Train real identities:",
        len(real_ids)
    )

    print(
        "Train fake identities:",
        len(fake_ids)
    )



    kept_real = 0
    kept_fake = 0



    # --------------------
    # Balance per identity
    # --------------------

    for identity in real_ids:


        if identity not in fake_ids:
            continue


        real_images = real_ids[identity]

        fake_images = fake_ids[identity]


        # number of real images
        n = len(real_images)


        # select same number fake images
        selected_fake = random.sample(
            fake_images,
            min(n, len(fake_images))
        )


        # copy real

        for img in real_images:

            shutil.copy2(
                os.path.join(real_dir,img),
                os.path.join(out_real,img)
            )

            kept_real += 1



        # copy fake

        for img in selected_fake:

            shutil.copy2(
                os.path.join(fake_dir,img),
                os.path.join(out_fake,img)
            )

            kept_fake += 1



    print("\nBalanced training result")
    print("-----------------------")
    print("Real images:", kept_real)
    print("Fake images:", kept_fake)




# ==========================
# Run
# ==========================


print("Balancing training set...")
balance_train()


print("\nCopying validation...")
copy_folder("val")


print("Copying test...")
copy_folder("test")


print("\nFinished")
