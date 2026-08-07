import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import random
import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


from models.hybrid_model import HybridModel
from transforms.dwt_transform import DWTTransform
from transforms.dual_transform import DualTransform



# =====================================================
# ARGUMENTS
# =====================================================

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run", type=int, default=1)

args = parser.parse_args()


SEED = args.seed
RUN = args.run



# =====================================================
# RANDOM SEED
# =====================================================

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# =====================================================
# CONFIGURATION
# =====================================================

DATA_VARIANT = "ALL"
IS_GRAY = False


data_root = "/home/amna/Final_Trained/DFFD/This_is_final_DFFD_A_dataset_balanced/"


exp_root = (
    f"/home/amna/projects/dwt-ext/results/"
    f"80-5-15/MM/WOA/DFFD/"
    f"seed_{SEED}/run_{RUN}"
)



checkpoint_dir = os.path.join(exp_root, "checkpoints")
log_dir = os.path.join(exp_root, "logs")
report_dir = os.path.join(exp_root, "reports")


os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)



latest_checkpoint = os.path.join(
    checkpoint_dir,
    "latest_checkpoint.pth"
)


best_model_path = os.path.join(
    checkpoint_dir,
    "best_model.pth"
)



train_log_path = os.path.join(
    log_dir,
    "training_log.csv"
)



device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("Using device:", device)



# =====================================================
# DATASET
# =====================================================


train_dir = os.path.join(data_root, "Train")
val_dir = os.path.join(data_root, "Val")



dwt_transform = DWTTransform(
    variant=DATA_VARIANT,
    is_gray=IS_GRAY
)


transform = DualTransform(
    dwt_transform
)



train_ds = datasets.ImageFolder(
    train_dir,
    transform=transform
)


val_ds = datasets.ImageFolder(
    val_dir,
    transform=transform
)



train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=20,
    pin_memory=True
)


val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=20,
    pin_memory=True
)



print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))
print("Batches per epoch:", len(train_loader))



# =====================================================
# MODEL
# =====================================================


model = HybridModel(
    "/home/amna/projects/dwt-ext/"
    "mix-dataset-transformer_resnet50.pth"
).to(device)



criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)



optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)



scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)



num_epochs = 30



# =====================================================
# RESUME TRAINING
# =====================================================


start_epoch = 0
best_val_acc = 0.0
train_log = []



if os.path.exists(latest_checkpoint):

    print("\nCheckpoint found.")
    print("Resuming training...\n")


    checkpoint = torch.load(
        latest_checkpoint,
        map_location=device
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )


    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )


    start_epoch = checkpoint["epoch"]


    best_val_acc = checkpoint["best_val_acc"]


    if os.path.exists(train_log_path):

        train_log = pd.read_csv(
            train_log_path
        ).to_dict("records")



    print(
        f"Resumed from epoch {start_epoch}"
    )

    print(
        f"Best validation accuracy: {best_val_acc:.4f}"
    )



else:

    print("\nNo checkpoint found.")
    print("Starting new training.\n")



# =====================================================
# TRAINING
# =====================================================


start_time = time.time()



for epoch in range(
    start_epoch,
    num_epochs
):


    print(
        f"\n===== Epoch {epoch+1}/{num_epochs} ====="
    )



    # ---------------- TRAIN ----------------

    model.train()


    train_loss = 0
    train_correct = 0
    total = 0



    for batch_idx, ((rgb, dwt), labels) in enumerate(train_loader):


        rgb = rgb.to(device, non_blocking=True)
        dwt = dwt.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)



        optimizer.zero_grad()



        logits = model(
            rgb,
            dwt
        )


        loss = criterion(
            logits,
            labels
        )


        loss.backward()


        optimizer.step()



        train_loss += loss.item()


        preds = torch.argmax(
            logits,
            dim=1
        )


        train_correct += (
            preds == labels
        ).sum().item()


        total += labels.size(0)



        if batch_idx % 50 == 0:

            print(
                f"Batch {batch_idx}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )



    train_acc = train_correct / total

    train_loss_avg = (
        train_loss / len(train_loader)
    )



    # ---------------- VALIDATION ----------------


    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    for (rgb, dwt), labels in val_loader:

        rgb = rgb.to(device)
        dwt = dwt.to(device)
        labels = labels.to(device)

        logits = model(rgb, dwt)
        loss = criterion(logits, labels)

        val_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        val_correct += (preds == labels).sum().item()
        val_total += labels.size(0)




    val_loss_avg = (
        val_loss / len(val_loader)
    )


    val_acc = (
        val_correct / val_total
    )



    scheduler.step(
        val_loss_avg
    )



    print(
        f"Epoch Completed | "
        f"Train Loss: {train_loss_avg:.4f} | "
        f"Val Loss: {val_loss_avg:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )



    # LOG

    train_log.append(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss_avg,
            "val_loss": val_loss_avg,
            "train_acc": train_acc,
            "val_acc": val_acc
        }
    )



    # SAVE LOG EVERY EPOCH

    pd.DataFrame(
        train_log
    ).to_csv(
        train_log_path,
        index=False
    )



    # UPDATE BEST

    if val_acc > best_val_acc:

        best_val_acc = val_acc


        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc
            },
            best_model_path
        )


        print(
            "Best model saved."
        )



    # SAVE CHECKPOINT

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc
        },
        latest_checkpoint
    )


    print(
        "Checkpoint saved."
    )



# =====================================================
# COMBINED TRAINING CURVE PLOT
# =====================================================


log_df = pd.DataFrame(train_log)


fig, ax_loss = plt.subplots(figsize=(10, 6))


ax_loss.plot(
    log_df["epoch"],
    log_df["train_loss"],
    label="Train Loss",
    color="tab:blue"
)

ax_loss.plot(
    log_df["epoch"],
    log_df["val_loss"],
    label="Val Loss",
    color="tab:blue",
    linestyle="--"
)

ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss", color="tab:blue")
ax_loss.tick_params(axis="y", labelcolor="tab:blue")


ax_acc = ax_loss.twinx()

ax_acc.plot(
    log_df["epoch"],
    log_df["train_acc"],
    label="Train Acc",
    color="tab:orange"
)

ax_acc.plot(
    log_df["epoch"],
    log_df["val_acc"],
    label="Val Acc",
    color="tab:orange",
    linestyle="--"
)

ax_acc.set_ylabel("Accuracy", color="tab:orange")
ax_acc.tick_params(axis="y", labelcolor="tab:orange")


lines_1, labels_1 = ax_loss.get_legend_handles_labels()
lines_2, labels_2 = ax_acc.get_legend_handles_labels()

ax_loss.legend(
    lines_1 + lines_2,
    labels_1 + labels_2,
    loc="center right"
)


plt.title("Training / Validation Loss & Accuracy")
fig.tight_layout()


plot_path = os.path.join(
    report_dir,
    "training_curves.png"
)

plt.savefig(plot_path)
plt.close(fig)


print(
    f"Training curves plot saved to {plot_path}"
)



# =====================================================
# FINAL REPORT
# =====================================================


summary = {

    "seed": SEED,

    "run": RUN,

    "best_val_acc": best_val_acc,

    "training_time_sec":
        round(
            time.time() - start_time,
            2
        ),

    "variant": DATA_VARIANT,

    "is_gray": IS_GRAY

}



with open(
    os.path.join(
        report_dir,
        "training_summary.json"
    ),
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )



print("\nTraining completed successfully.")
print(
    "Best validation accuracy:",
    best_val_acc
)
