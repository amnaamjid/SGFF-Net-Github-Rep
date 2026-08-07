import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from drn import DRN, validate_step, save_model

# ========================== CONFIG ==========================
DATA_DIR = "/home5/amna.seecs/MS/Deepfake/Dataset/imagenette2-160"
SAVE_DIR = "/home5/amna.seecs/MS/Deepfake/Code/Models/DRN/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 7  # Early stopping patience (around epoch 30-34 stabilization)

# ========================== DATA ==========================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
val_data   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

# ========================== MODEL ==========================
model = DRN(in_ch=3, num_classes=len(train_data.classes)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Scheduler based on VALIDATION LOSS (per paper)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

best_val_acc = 0.0
best_val_loss = float('inf')
best_epoch = 0
epochs_no_improve = 0

train_losses, val_losses, val_accuracies = [], [], []

# ========================== TRAINING LOOP ==========================
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total

    val_loss, val_acc = validate_step(model, val_loader, criterion, DEVICE)
    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Track best model (by validation loss, as per paper)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        best_epoch = epoch
        epochs_no_improve = 0
        save_model(model, os.path.join(SAVE_DIR, "drn_imagenette_best.pth"))
    else:
        epochs_no_improve += 1

    # Early stopping if validation stagnates
    if epochs_no_improve >= PATIENCE:
        print(f"\n⚠️ Validation loss has not improved for {PATIENCE} epochs. Early stopping at epoch {epoch}.")
        break

# ========================== PLOTS ==========================
epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(12,5))

# ---- Loss Plot ----
plt.subplot(1,2,1)
plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
plt.plot(epochs_range, val_losses, label='Validation Loss', marker='o')
plt.axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Training & Validation Loss (DRN Imagenette)')
plt.legend()

# ---- Accuracy Plot ----
plt.subplot(1,2,2)
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', color='green', marker='o')
plt.axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.title('Validation Accuracy (DRN Imagenette)')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_curves_best_marked.png"))
plt.close()

# ========================== SUMMARY ==========================
print("\n========== TRAINING SUMMARY ==========")
print(f"✅ Training complete.")
print(f"🏆 Best Epoch: {best_epoch}")
print(f"📈 Best Val Accuracy: {best_val_acc:.4f}")
print(f"📉 Best Val Loss: {best_val_loss:.4f}")
print(f"💾 Best Model Saved: {os.path.join(SAVE_DIR, 'drn_imagenette_best.pth')}")
print(f"📊 Graph Saved: {os.path.join(SAVE_DIR, 'training_curves_best_marked.png')}")
