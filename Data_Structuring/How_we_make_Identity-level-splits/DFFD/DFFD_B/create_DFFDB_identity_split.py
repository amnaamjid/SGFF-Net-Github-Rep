import os
import shutil
import random


REAL_DIR = "DFFD_B"
FAKE_DIR = "stylegan_selected"

OUTPUT = "DFFD_B_eval_balanced"


random.seed(42)


REAL_OUT = os.path.join(
    OUTPUT,
    "real"
)

FAKE_OUT = os.path.join(
    OUTPUT,
    "fake"
)


os.makedirs(REAL_OUT, exist_ok=True)
os.makedirs(FAKE_OUT, exist_ok=True)



# ==========================
# Read real images
# ==========================

real_images = {}


for img in os.listdir(REAL_DIR):

    if img.lower().endswith(".jpg"):

        name = os.path.splitext(img)[0]

        identity = "_".join(
            name.split("_")[1:]
        )


        real_images.setdefault(
            identity,
            []
        ).append(img)



# ==========================
# Balance identity-wise
# ==========================


total_real = 0
total_fake = 0

valid_identity = 0



for identity in real_images:


    fake_folder = os.path.join(
        FAKE_DIR,
        identity
    )


    if not os.path.exists(fake_folder):
        continue



    real_list = real_images[identity]

    fake_list = [
        x for x in os.listdir(fake_folder)
        if x.lower().endswith(
            (".jpg",".png")
        )
    ]


    if len(fake_list)==0:
        continue



    # balance number

    keep = min(
        len(real_list),
        len(fake_list)
    )


    if keep == 0:
        continue



    valid_identity += 1


    random.shuffle(real_list)
    random.shuffle(fake_list)



    real_keep = real_list[:keep]
    fake_keep = fake_list[:keep]



    # copy real

    for img in real_keep:

        shutil.copy2(
            os.path.join(
                REAL_DIR,
                img
            ),
            os.path.join(
                REAL_OUT,
                img
            )
        )

        total_real += 1



    # copy fake

    for img in fake_keep:

        shutil.copy2(
            os.path.join(
                fake_folder,
                img
            ),
            os.path.join(
                FAKE_OUT,
                img
            )
        )

        total_fake += 1




print("======================")
print("Balanced Dataset Done")
print("======================")

print("Identities used:", valid_identity)

print("Real images:", total_real)

print("Fake images:", total_fake)

print("Total images:", total_real+total_fake)