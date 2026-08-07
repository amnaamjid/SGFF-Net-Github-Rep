import os
import shutil
import random
import pandas as pd


# ============================
# Paths
# ============================

REAL_DIR = "DFFD_A"

PGAN1_DIR = "pggan_v1_selected"
PGAN2_DIR = "pggan_v2_selected"


OUTPUT = "DFFD_A_dataset"


SEED = 42


# ============================
# Split ratio
# ============================

TRAIN_RATIO = 0.80
VAL_RATIO = 0.05
TEST_RATIO = 0.15



random.seed(SEED)



# ============================
# Get real identities
# ============================

print("Reading real identities...")


real_images = {}

for img in os.listdir(REAL_DIR):

    if img.endswith(".jpg"):

        name = os.path.splitext(img)[0]

        identity = "_".join(
            name.split("_")[1:]
        )

        real_images.setdefault(identity, []).append(img)



print("Real identities:", len(real_images))



# ============================
# Find identities having fake
# ============================

print("Checking fake availability...")


valid_ids = []


for identity in real_images:


    fake_exists = False


    if os.path.exists(
        os.path.join(PGAN1_DIR, identity)
    ):
        if len(os.listdir(
            os.path.join(PGAN1_DIR, identity)
        )) > 0:
            fake_exists = True



    if os.path.exists(
        os.path.join(PGAN2_DIR, identity)
    ):
        if len(os.listdir(
            os.path.join(PGAN2_DIR, identity)
        )) > 0:
            fake_exists = True



    if fake_exists:
        valid_ids.append(identity)



print("Identities with real + fake:", len(valid_ids))



# ============================
# Identity split
# ============================


random.shuffle(valid_ids)


total = len(valid_ids)


train_end = int(total * TRAIN_RATIO)

val_end = train_end + int(total * VAL_RATIO)



train_ids = valid_ids[:train_end]

val_ids = valid_ids[train_end:val_end]

test_ids = valid_ids[val_end:]



print("\nSplit identities")

print("Train:", len(train_ids))

print("Val:", len(val_ids))

print("Test:", len(test_ids))



# ============================
# Copy function
# ============================


def copy_identity(identity, split):


    real_out = os.path.join(
        OUTPUT,
        split,
        "real"
    )

    fake_out = os.path.join(
        OUTPUT,
        split,
        "fake"
    )


    os.makedirs(real_out, exist_ok=True)

    os.makedirs(fake_out, exist_ok=True)



    # -------- REAL --------

    for img in real_images[identity]:

        shutil.copy2(
            os.path.join(
                REAL_DIR,
                img
            ),
            os.path.join(
                real_out,
                img
            )
        )



    # -------- PGAN v1 --------

    p1 = os.path.join(
        PGAN1_DIR,
        identity
    )


    if os.path.exists(p1):

        for img in os.listdir(p1):

            shutil.copy2(
                os.path.join(p1,img),
                os.path.join(fake_out,img)
            )



    # -------- PGAN v2 --------


    p2 = os.path.join(
        PGAN2_DIR,
        identity
    )


    if os.path.exists(p2):

        for img in os.listdir(p2):

            shutil.copy2(
                os.path.join(p2,img),
                os.path.join(fake_out,img)
            )



# ============================
# Create dataset
# ============================


print("\nCopying dataset...")


for i in train_ids:
    copy_identity(i,"train")


for i in val_ids:
    copy_identity(i,"val")


for i in test_ids:
    copy_identity(i,"test")



print("\nFinished")
