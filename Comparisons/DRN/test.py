import os
import time
import sys
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image

# === Add your DRN model path ===
sys.path.append("/home/amna/projects/Comparison/DRN/Models/DRN")
from drn import DRN

# ============================================================== #
# CONFIG
# ============================================================== #
MODE = "grayscale_fft"   # <--- ADD THIS LINE



exp_root = "/home/amna/projects/Comparison/DRN/Seed_29"
model_path = os.path.join(exp_root, "checkpoints/best_model.pth")

datasets_root = {
    "DFFD_A": "/home/amna/projects/Comparison/DRN/Grayscale/DFFD_Test",
    "DFF_C": "/home/amna/projects/Comparison/DRN/Grayscale/DFF_C_Test",
    "DiffFace_A": "/home/amna/projects/Comparison/DRN/Grayscale/DiffFace_A_Test",
    "DiffFace_B": "//home/amna/projects/Comparison/DRN/Grayscale/DiffFace_B_Test",
    "DiffFace_C": "/home/amna/projects/Comparison/DRN/Grayscale/DiffFace_C_Test",
    "DiffFace_D": "/home/amna/projects/Comparison/DRN/Grayscale/DiffFace_D_Test",
    "DiffFace_E": "/home/amna/projects/Comparison/DRN/Grayscale/DiffFace_E_Test"

}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

# ============================================================== #
# HELPER FUNCTION: High-Frequency FFT Transform
# ============================================================== #
def high_frequency_image(pil_img, hpf_radius=30, resize=(224, 224)):
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

# ============================================================== #
# TRANSFORMS (auto-switch)
# ============================================================== #
transform = transforms.Compose([
transforms.Lambda(lambda img: high_frequency_image(img, hpf_radius=30, resize=(224, 224))),transforms.ToTensor()])
in_ch = 1


# ============================================================== #
# LOAD MODEL
# ============================================================== #
model = DRN(in_ch=in_ch, num_classes=2, dropout=0.25).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# ============================================================== #
# EVALUATION FUNCTION
# ============================================================== #

def evaluate_dataset(name, data_dir):
    print(f"\n Evaluating on {name} ...")
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=64)

    all_labels, all_preds, all_probs = [], [], []
    start_time = time.time()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    inference_time = time.time() - start_time
    avg_inference_time = inference_time / len(dataset)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    prec_curve, rec_curve, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(rec_curve, prec_curve)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    # === SAVE RAW DATA FOR LATER PLOTTING ===
    save_dir = os.path.join(exp_root, "raw_outputs")
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, f"{name}_{MODE}_labels.npy"), np.array(all_labels))
    np.save(os.path.join(save_dir, f"{name}_{MODE}_probs.npy"), np.array(all_probs))
    np.save(os.path.join(save_dir, f"{name}_{MODE}_preds.npy"), np.array(all_preds))
    
    

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title(f"Confusion Matrix - {name} ({MODE.upper()})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(exp_root, f"reports/confusion_matrix_{name}_{MODE}.png"))
    plt.close()

    # ROC Curve
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title(f"ROC Curve - {name} ({MODE.upper()})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(os.path.join(exp_root, f"reports/roc_curve_{name}_{MODE}.png"))
    plt.close()

    # PR Curve
    plt.figure(figsize=(6, 6))
    plt.plot(rec_curve, prec_curve, color='green', label=f'PR (AUC = {pr_auc:.3f})')
    plt.title(f"Precision-Recall Curve - {name} ({MODE.upper()})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(os.path.join(exp_root, f"reports/pr_curve_{name}_{MODE}.png"))
    plt.close()

    # Fixed indentation
    metrics = {
        "Dataset": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "PR-AUC": round(pr_auc, 4),
        "EER ↓": round(eer, 4),
        "Total Inference Time (sec)": round(inference_time, 2),
        "Avg Inf/Image (sec)": round(avg_inference_time, 5),
        "Samples": len(dataset),
    }
    
    

    return metrics

# ============================================================== #
# MAIN EVALUATION LOOP
# ============================================================== #
results = []
for name, path in datasets_root.items():
    if os.path.exists(path):
        metrics = evaluate_dataset(name, path)
        results.append(metrics)
    else:
        print(f" Dataset path not found: {path}")

# ============================================================== #
# SAVE RESULTS
# ============================================================== #
report_dir = os.path.join(exp_root, "reports")
os.makedirs(report_dir, exist_ok=True)

report_path = os.path.join(report_dir, f"evaluation_results_{MODE}.csv")
pd.DataFrame(results).to_csv(report_path, index=False)
json.dump(results, open(os.path.join(report_dir, f"evaluation_results_{MODE}.json"), "w"), indent=4)

print("\n Evaluation complete. Results saved to:")
print(f"{report_path}")
