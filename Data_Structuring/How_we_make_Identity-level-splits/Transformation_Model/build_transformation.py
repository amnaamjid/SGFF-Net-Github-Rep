import shutil
from pathlib import Path

ROOT = Path("/home/amna/Ready_to_send_Dataset/DFFD")

REAL_DIR = ROOT / "Celeb-HQ-Real"

GENERATORS = [
    "ADM",
    "DDIM",
    "DDPM",
    "LDM",
    "pggan_v1",
    "pggan_v2",
    "stylegan",
]

OUT_REAL = ROOT / "Final_DFFD/Real"
OUT_FAKE = ROOT / "Final_DFFD/Fake"

OUT_REAL.mkdir(parents=True, exist_ok=True)
OUT_FAKE.mkdir(parents=True, exist_ok=True)


def get_real_id(name):
    return Path(name).stem.split("_")[0].lstrip("0")


def get_fake_id(name, gen):
    stem = Path(name).stem

    if gen in ["ADM", "DDIM", "DDPM", "LDM"]:
        return stem.split("_")[0].lstrip("0")

    else:
        return stem.split("_")[-1].lstrip("0")


# -----------------------------
# Build index
# -----------------------------

index = {}

for gen in GENERATORS:

    idx = {}

    for img in (ROOT / gen).glob("*.jpg"):

        fid = get_fake_id(img.name, gen)

        idx.setdefault(fid, []).append(img)

    index[gen] = idx


count = {g: 0 for g in GENERATORS}
used = set()

copied = 0
missing = 0

# -----------------------------
# Process
# -----------------------------

for real in sorted(REAL_DIR.glob("*.jpg")):

    rid = get_real_id(real.name)

    candidates = []

    for gen in GENERATORS:

        if rid not in index[gen]:
            continue

        for fake in index[gen][rid]:

            if fake in used:
                continue

            candidates.append((count[gen], gen, fake))
            break

    if len(candidates) == 0:
        missing += 1
        continue

    # choose generator having minimum current count
    candidates.sort()

    _, gen, fake = candidates[0]

    shutil.copy2(real, OUT_REAL / real.name)
    shutil.copy2(fake, OUT_FAKE / f"{gen}_{fake.name}")

    used.add(fake)
    count[gen] += 1
    copied += 1


print("\n========== SUMMARY ==========\n")

for g in GENERATORS:
    print(f"{g:10s}: {count[g]}")

print("----------------------------")
print("Pairs :", copied)
print("Missing:", missing)