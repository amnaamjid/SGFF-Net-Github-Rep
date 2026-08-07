import torch
import sys
from torchsummary import summary

# === Add your DRN model path ===
sys.path.append("/home5/amna.seecs/MS/Deepfake/Code/Models/DRN")

# === Import your DRN model ===
from drn import DRN  # make sure the class name in drn.py is 'DRN'

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"? Using device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")

# === Initialize model ===
try:
    model = DRN(num_classes=2).to(device)  # adjust if your constructor differs
except TypeError:
    # if your DRN doesn't take num_classes arg, try default init
    model = DRN().to(device)

model.eval()
print("\n? Model loaded successfully on GPU")

# === Print model summary (optional, can be skipped if large) ===
try:
    summary(model, (3, 224, 224))
except Exception as e:
    print(f"?? Could not print summary: {e}")

# === Dummy input like your RGB images (batch of 8) ===
x = torch.randn(8, 3, 224, 224).to(device)

# === Forward pass ===
with torch.no_grad():
    y = model(x)

print("\n? Forward pass successful!")
print(f"Output shape: {y.shape}")

import torch
from drn import DRN   # or whatever your class name is

model = DRN()         # initialize the model
print(model)

