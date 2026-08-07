import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

import random

SEED = 29

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ======================================
# PATH CONFIG
# ======================================
sys.path.append("/home/amna/projects/Comparison/DRN/Models/DRN")
from drn import DRN

data_root = "/home/amna/projects/Comparison/This_is_final_DFF_C/Grayscale"
exp_root = "/home/amna/projects/Comparison/DRN/Seed_29"


train_dir = os.path.join(data_root, "Train")
val_dir = os.path.join(data_root, "Val")

os.makedirs(os.path.join(exp_root, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(exp_root, "plots"), exist_ok=True)
os.makedirs(os.path.join(exp_root, "logs"), exist_ok=True)
os.makedirs(os.path.join(exp_root, "reports"), exist_ok=True)

# ======================================
# DEVICE
# ======================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")


# ======================================
# FFT High-Frequency Function
# ======================================
def high_frequency_image(pil_img, hpf_radius=30, resize=(224,224)):
    """
    Convert a PIL grayscale image to high-frequency emphasized version using FFT.
    """
    img = pil_img.convert("L")
    img_np = np.array(img, dtype=np.float32)
    
    # FFT and shift
    f = np.fft.fft2(img_np)
    fshift = np.fft.fftshift(f)
    
    # Create high-pass mask
    rows, cols = img_np.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.float32)
    mask[crow - hpf_radius:crow + hpf_radius, ccol - hpf_radius:ccol + hpf_radius] = 0
    
    # Apply mask
    fshift_filtered = fshift * mask
    
    # Inverse FFT
    img_back = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(img_back)
    img_back = np.abs(img_back)
    
    # Normalize to [0,255]
    img_back = (img_back - img_back.min()) / (img_back.max() - img_back.min() + 1e-8)
    img_back = (img_back * 255).astype(np.uint8)
    
    # Convert back to PIL + resize
    img_out = Image.fromarray(img_back)
    if resize is not None:
        img_out = img_out.resize(resize)
    return img_out

# ======================================
# TRANSFORMS
# ======================================
transform = transforms.Compose([
transforms.Lambda(lambda img: high_frequency_image(img, hpf_radius=30, resize=(224,224))),
transforms.ToTensor() ])

# ======================================
# DATA LOADERS
# ======================================
train_ds = datasets.ImageFolder(train_dir, transform=transform)
val_ds = datasets.ImageFolder(val_dir, transform=transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=16)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=16)

# ======================================
# MODEL
# ======================================
model = DRN(in_ch=1, num_classes=2, dropout=0.25).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

# ======================================
# TRAIN LOOP
# ======================================
num_epochs = 30
best_val_acc = 0
train_log = []
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct, total = 0, 0, 0
    
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = train_correct / total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    scheduler.step(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}")
    train_log.append({
        "epoch": epoch+1,
        "train_loss": train_loss/len(train_loader),
        "val_loss": val_loss/len(val_loader),
        "train_acc": train_acc,
        "val_acc": val_acc
    })

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(exp_root, "checkpoints/best_model.pth"))

# ======================================
# LOGGING & PLOTS
# ======================================
train_df = pd.DataFrame(train_log)
train_df.to_csv(os.path.join(exp_root, "logs/training_log.csv"), index=False)

plt.figure(figsize=(10,5))
plt.plot(train_df["epoch"], train_df["train_acc"], label="Train Acc")
plt.plot(train_df["epoch"], train_df["val_acc"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.savefig(os.path.join(exp_root, "plots/accuracy_curve.png"))

plt.figure(figsize=(10,5))
plt.plot(train_df["epoch"], train_df["train_loss"], label="Train Loss")
plt.plot(train_df["epoch"], train_df["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.savefig(os.path.join(exp_root, "plots/loss_curve.png"))

summary = {
    "best_val_acc": best_val_acc,
    "training_time_sec": round(time.time() - start_time, 2),
    "num_epochs": num_epochs,
    "device": str(device)
}
json.dump(summary, open(os.path.join(exp_root, "reports/training_summary.json"), "w"), indent=4)
