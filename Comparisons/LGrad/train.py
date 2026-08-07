import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import ImageFolder
from torchvision import transforms

from torch.utils.data import DataLoader

from model.lgrad_model import LGradClassifier


# ==========================================================
# CONFIG
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_path = "/home/amna/HPC_DATAGAN/Same-Training/RGB/Same-Training_Gradient/Train"
val_path   = "/home/amna/HPC_DATAGAN/Same-Training/RGB/Same-Training_Gradient/Val"

batch_size = 32
epochs = 30

lr = 1e-4
weight_decay = 1e-4

save_path = "LGrad_best.pth"


# ==========================================================
# DATA
# ==========================================================

train_transform = transforms.Compose([

    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()

])

val_transform = transforms.Compose([

    transforms.Resize((224,224)),
    transforms.ToTensor()

])


train_dataset = ImageFolder(
    train_path,
    transform=train_transform
)

val_dataset = ImageFolder(
    val_path,
    transform=val_transform
)


train_loader = DataLoader(

    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True

)


val_loader = DataLoader(

    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True

)


print("Training Images :", len(train_dataset))
print("Validation Images :", len(val_dataset))


# ==========================================================
# MODEL
# ==========================================================

model = LGradClassifier()

model = model.to(device)


# ==========================================================
# LOSS
# ==========================================================

criterion = nn.CrossEntropyLoss(

    label_smoothing=0.05

)


# ==========================================================
# OPTIMIZER
# ==========================================================

optimizer = optim.AdamW(

    model.parameters(),
    lr=lr,
    weight_decay=weight_decay

)


scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,
    mode="max",
    factor=0.5,
    patience=3,
    verbose=True

)


# ==========================================================
# TRAINING
# ==========================================================

best_acc = 0.0


for epoch in range(epochs):

    ##############################################
    # TRAIN
    ##############################################

    model.train()

    train_loss = 0

    train_correct = 0

    train_total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)

        loss = criterion(outputs, labels)


        loss.backward()

        optimizer.step()


        train_loss += loss.item()


        preds = outputs.argmax(1)

        train_correct += (preds == labels).sum().item()

        train_total += labels.size(0)


    train_acc = 100 * train_correct / train_total

    train_loss = train_loss / len(train_loader)



    ##############################################
    # VALIDATION
    ##############################################

    model.eval()


    val_loss = 0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(images)

            loss = criterion(outputs, labels)


            val_loss += loss.item()


            preds = outputs.argmax(1)

            val_correct += (preds == labels).sum().item()

            val_total += labels.size(0)



    val_acc = 100 * val_correct / val_total

    val_loss = val_loss / len(val_loader)



    ##############################################
    # Scheduler
    ##############################################

    scheduler.step(val_acc)



    ##############################################
    # Save Best
    ##############################################

    if val_acc > best_acc:

        best_acc = val_acc


        torch.save({

            "epoch":epoch+1,

            "model_state_dict":model.state_dict(),

            "optimizer_state_dict":optimizer.state_dict(),

            "best_acc":best_acc

        }, save_path)



    ##############################################
    # PRINT
    ##############################################

    print("-"*70)

    print(f"Epoch [{epoch+1}/{epochs}]")

    print(f"Train Loss : {train_loss:.4f}")

    print(f"Train Acc  : {train_acc:.2f}%")

    print(f"Val Loss   : {val_loss:.4f}")

    print(f"Val Acc    : {val_acc:.2f}%")

    print(f"Best Val   : {best_acc:.2f}%")

    print(f"Learning Rate : {optimizer.param_groups[0]['lr']:.6f}")



print("\nTraining Finished.")

print(f"Best Validation Accuracy : {best_acc:.2f}%")
