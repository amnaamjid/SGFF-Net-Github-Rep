import os
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

from scipy.optimize import brentq
from scipy.interpolate import interp1d

from model.lgrad_model import LGradClassifier


# ==========================================================
# SEED
# ==========================================================

seed = 20

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ==========================================================
# CONFIG
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = "LGrad_best.pth"

batch_size = 32

num_workers = 16


datasets_root = {

    "DFFD_Test": "/home/amna/projects/Comparisons/Compariosns/LGrad/LGrad/LGradComparisonsWithSGFFNET/Test/DFFD_Test/Gradient/",

    "DFF_C_Test": "/home/amna/projects/Comparisons/Compariosns/LGrad/LGrad/LGradComparisonsWithSGFFNET/Test/DFF_C_Test/Gradient/",

    "DiffFace_A": "/home/amna/projects/Comparisons/Compariosns/LGrad/LGrad/LGradComparisonsWithSGFFNET/Test/DiffFace_A/Gradient/"
}


# ==========================================================
# TRANSFORM
# ==========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ==========================================================
# LOAD MODEL
# ==========================================================

model = LGradClassifier()

state = torch.load(
    checkpoint,
    map_location=device
)

model.load_state_dict(state["model_state_dict"])

model.to(device)
model.eval()


results = []


# ==========================================================
# LOOP OVER DATASETS
# ==========================================================

for dataset_name, test_path in datasets_root.items():

    print(f"\nEvaluating {dataset_name}")

    if not os.path.exists(test_path):
        print(f"Path not found: {test_path}")
        continue

    test_dataset = ImageFolder(
        test_path,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print("Images:", len(test_dataset))

    # -------------------------------
    # Warmup
    # -------------------------------

    for _ in range(5):

        for images, _ in test_loader:

            images = images.to(device)

            with torch.no_grad():
                _ = model(images)

            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # -------------------------------
    # Evaluation
    # -------------------------------

    y_true = []
    y_pred = []
    y_score = []

    total_time = 0
    total_images = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.time()

            outputs = model(images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.time()

            total_time += end - start
            total_images += images.size(0)

            probs = F.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    # -------------------------------
    # Metrics
    # -------------------------------

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except:
        roc_auc = np.nan

    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)

        eer = brentq(
            lambda x: 1. - x - interp1d(fpr, tpr)(x),
            0.,
            1.
        )

    except:
        eer = np.nan

    latency = (total_time / total_images) * 1000

    results.append({

        "Dataset": dataset_name,
        "Images": len(test_dataset),

        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Recall (%)": round(recall * 100, 2),
        "F1 (%)": round(f1 * 100, 2),
        "ROC-AUC (%)": round(roc_auc * 100, 2) if not np.isnan(roc_auc) else np.nan,
        "EER (%)": round(eer * 100, 2) if not np.isnan(eer) else np.nan,

        "Latency (ms/image)": round(latency, 3)
    })

    print(f"Accuracy : {accuracy*100:.2f}%")
    print(f"F1       : {f1*100:.2f}%")
    print(f"EER      : {eer*100:.2f}%")
    print(f"Latency  : {latency:.3f} ms/image")


# ==========================================================
# SAVE RESULTS
# ==========================================================

results_df = pd.DataFrame(results)

results_df.to_excel(
    "LGrad_Test_Results.xlsx",
    index=False
)

print("\n========================================")
print(results_df)
print("========================================")
print("\nResults saved to: LGrad_Test_Results.xlsx")