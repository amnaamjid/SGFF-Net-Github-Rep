import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import time
import json
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, confusion_matrix
)


# ==============================================================
# 1. REPRODUCIBILITY
# ==============================================================

def set_seed(seed=6):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--run", type=int, required=True)

args = parser.parse_args()

SEED = args.seed
RUN = args.run

set_seed(SEED)


# ==============================================================
# 2. CONFIGURATION
# ==============================================================

DATA_VARIANT = "ALL"
IS_GRAY = False


exp_root = (
    f"/home/amna/projects/dwt-ext/results/"
    f"80-5-15/MM/WA/DFF_C/"
    f"seed_{SEED}/run_{RUN}"
)



model_path = os.path.join(
    exp_root,
    "checkpoints/best_model.pth"
)

report_dir = os.path.join(
    exp_root,
    "reports"
)

raw_dir = os.path.join(
    exp_root,
    "raw_outputs"
)

# NEW:
# Prediction-level CSV files for detailed future analysis in R
prediction_dir = os.path.join(
    exp_root,
    "predictions"
)

os.makedirs(
    report_dir,
    exist_ok=True
)

os.makedirs(
    raw_dir,
    exist_ok=True
)

# NEW:
os.makedirs(
    prediction_dir,
    exist_ok=True
)


datasets_root = {

    "MIX_ALL": "/home/amna/HPC_DATA/MIX_ALL/Test/",
    "DFF_A": "/home/amna/HPC_DATA/DFF/DFF_A/80-5-15/Test",
    "DFF_B": "/home/amna/HPC_DATA/DFF/DFF_B/80-5-15/Test",
    "DFF_C": "/home/amna/HPC_DATA/DFF/DFF_C/80-5-15/Test/",
    "DFFD_A": "/home/amna/Ready_to_send_Dataset/DFFD/DFFDA/This_is_final_DFFD_A_dataset_balanced/Test/",
    "DFFD_B": "/home/amna/Ready_to_send_Dataset/DFFD/DFFDB/This_is_final_DFFD_B_eval_balanced",
    "DiffFace_A": "/home/amna/Ready_to_send_Dataset/DiffFacee/This_is_final_DiffFace/DiffFace_A/",
    "DiffFace_B": "/home/amna/Ready_to_send_Dataset/DiffFacee/This_is_final_DiffFace/DiffFace_B/",
    "DiffFace_C": "/home/amna/Ready_to_send_Dataset/DiffFacee/This_is_final_DiffFace/DiffFace_C/",
    "DiffFace_D": "/home/amna/Ready_to_send_Dataset/DiffFacee/This_is_final_DiffFace/DiffFace_D/",
    "DiffFace_E": "/home/amna/Ready_to_send_Dataset/DiffFacee/This_is_final_DiffFace/DiffFace_E/"
}


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}")


# ==============================================================
# 3. IMPORT YOUR PIPELINE
# ==============================================================

sys.path.append(
    "/home/amna/projects/dwt-ext/"
)

from models.hybrid_model import HybridModel
from transforms.dwt_transform import DWTTransform
from transforms.dual_transform import DualTransform


# ==============================================================
# 4. TRANSFORMS (MUST MATCH TRAINING)
# ==============================================================

dwt_transform = DWTTransform(
    variant=DATA_VARIANT,
    is_gray=IS_GRAY
)

transform = DualTransform(
    dwt_transform
)


# ==============================================================
# 5. LOAD MODEL
# ==============================================================

PRETRAIN_PATH = os.path.join(
    PROJECT_ROOT,
    "mix-dataset-transformer_resnet50.pth"
)

model = HybridModel(
    PRETRAIN_PATH
).to(device)

checkpoint = torch.load(
    model_path,
    map_location=device
)

model.load_state_dict(checkpoint)

model.eval()

print(
    "HybridModel loaded successfully.\n"
)


# ==============================================================
# 6. EVALUATION FUNCTION
# ==============================================================

def evaluate_dataset(
    name,
    data_dir
):

    print(
        f"Evaluating on {name} ..."
    )

    dataset = datasets.ImageFolder(
        data_dir,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=20,
        pin_memory=True
    )

    all_labels = []
    all_preds = []
    all_probs = []

    # NEW:
    # Store original image paths.
    # ImageFolder provides paths in exactly the same
    # order as the dataset samples.
    all_image_paths = []

    # ==============================================================
    # GPU WARM-UP (10 batches)
    # ==============================================================

    print(
        "Performing GPU warm-up..."
    )

    for i, ((rgb, dwt), labels) in enumerate(loader):

        if i >= 10:
            break

        rgb = rgb.to(
            device
        ).requires_grad_(True)

        dwt = dwt.to(
            device
        )

        _ = model(
            rgb,
            dwt
        )

    if device.type == "cuda":

        torch.cuda.synchronize()


    # ==============================================================
    # START TIMING
    # ==============================================================

    if device.type == "cuda":

        torch.cuda.synchronize()

    start_time = time.time()


    for i, ((rgb, dwt), labels) in enumerate(loader):

        rgb = rgb.to(
            device
        ).requires_grad_(True)

        dwt = dwt.to(
            device
        )

        labels = labels.to(
            device
        )

        outputs = model(
            rgb,
            dwt
        )

        outputs_detached = outputs.detach()

        probs = torch.softmax(
            outputs_detached,
            dim=1
        )[:, 1]

        preds = torch.argmax(
            outputs_detached,
            dim=1
        )


        all_labels.extend(
            labels.cpu().numpy()
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_probs.extend(
            probs.cpu().numpy()
        )

        # NEW:
        # Save exact image paths corresponding to each prediction.
        start_idx = i * loader.batch_size
        end_idx = min(
            start_idx + len(labels),
            len(dataset)
        )

        batch_paths = [
            dataset.samples[j][0]
            for j in range(
                start_idx,
                end_idx
            )
        ]

        all_image_paths.extend(
            batch_paths
        )


        if (i + 1) % 10 == 0:

            print(
                f"  Processed {i+1}/{len(loader)} batches",
                end="\r"
            )

        del outputs


    if device.type == "cuda":

        torch.cuda.synchronize()


    end_time = time.time()


    inference_time = (
        end_time -
        start_time
    )

    avg_time = (
        inference_time /
        len(dataset)
    )


    # ==============================================================
    # METRICS
    # ==============================================================

    acc = accuracy_score(
        all_labels,
        all_preds
    )

    prec = precision_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    rec = recall_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        zero_division=0
    )


    fpr, tpr, _ = roc_curve(
        all_labels,
        all_probs
    )

    roc_auc = auc(
        fpr,
        tpr
    )


    prec_curve, rec_curve, _ = precision_recall_curve(
        all_labels,
        all_probs
    )

    pr_auc = (
        auc(
            rec_curve,
            prec_curve
        )
        if len(rec_curve) > 1
        else 0.0
    )


    fnr = 1 - tpr

    eer_idx = np.nanargmin(
        np.abs(
            fnr -
            fpr
        )
    )

    eer = (
        fpr[eer_idx]
        if eer_idx < len(fpr)
        else np.nan
    )


    # ==============================================================
    # SAVE RAW OUTPUTS
    # ==============================================================

    np.save(
        os.path.join(
            raw_dir,
            f"{name}_labels.npy"
        ),
        np.array(
            all_labels
        )
    )

    np.save(
        os.path.join(
            raw_dir,
            f"{name}_probs.npy"
        ),
        np.array(
            all_probs
        )
    )

    np.save(
        os.path.join(
            raw_dir,
            f"{name}_preds.npy"
        ),
        np.array(
            all_preds
        )
    )


    # ==============================================================
    # NEW: SAVE COMPLETE PREDICTION-LEVEL CSV
    # ==============================================================
    #
    # This does NOT change your evaluation logic.
    #
    # It simply saves the information that you already calculated
    # in a format that R can directly read later.
    #
    # This file is extremely useful for:
    #
    # 1. Combined ROC-AUC
    # 2. Combined PR-AUC
    # 3. Confusion matrices
    # 4. False Positive analysis
    # 5. False Negative analysis
    # 6. Correct/incorrect predictions
    # 7. Confidence analysis
    # 8. Per-image error analysis
    # 9. Cross-dataset comparisons
    # ==============================================================

    prediction_df = pd.DataFrame({

        "experiment_train_dataset": "DFF_C",

        "seed": SEED,

        "run": RUN,

        "test_dataset": name,

        "image_path": all_image_paths,

        "true_label": np.array(
            all_labels
        ),

        "predicted_label": np.array(
            all_preds
        ),

        "prob_fake": np.array(
            all_probs
        ),

        "prob_real": (
            1 -
            np.array(
                all_probs
            )
        )

    })


    # NEW:
    # Correct / Incorrect prediction

    prediction_df[
        "correct"
    ] = (

        prediction_df[
            "true_label"
        ]

        ==

        prediction_df[
            "predicted_label"
        ]

    )


    # NEW:
    # Human-readable labels

    prediction_df[
        "true_class"
    ] = prediction_df[
        "true_label"
    ].map({

        0: "Real",

        1: "Fake"

    })


    prediction_df[
        "predicted_class"
    ] = prediction_df[
        "predicted_label"
    ].map({

        0: "Real",

        1: "Fake"

    })


    # NEW:
    # Error type for detailed error analysis

    prediction_df[
        "error_type"
    ] = "Correct"


    prediction_df.loc[

        (
            prediction_df[
                "true_label"
            ] == 0
        )

        &

        (
            prediction_df[
                "predicted_label"
            ] == 1
        ),

        "error_type"

    ] = "False Positive"


    prediction_df.loc[

        (
            prediction_df[
                "true_label"
            ] == 1
        )

        &

        (
            prediction_df[
                "predicted_label"
            ] == 0
        ),

        "error_type"

    ] = "False Negative"


    prediction_csv_path = os.path.join(

        prediction_dir,

        f"{name}_predictions.csv"

    )


    prediction_df.to_csv(

        prediction_csv_path,

        index=False

    )


    print(
        f"\nPrediction-level results saved: "
        f"{prediction_csv_path}"
    )


    # ==============================================================
    # CONFUSION MATRIX
    # ==============================================================

    cm = confusion_matrix(
        all_labels,
        all_preds
    )


    plt.figure(
        figsize=(5, 5)
    )


    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=[
            "Real",
            "Fake"
        ],

        yticklabels=[
            "Real",
            "Fake"
        ]

    )


    plt.title(
        f"Confusion Matrix - {name}"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "True"
    )

    plt.tight_layout()


    plt.savefig(

        os.path.join(

            report_dir,

            f"confusion_matrix_{name}.png"

        )

    )


    plt.close()


    # ==============================================================
    # ROC
    # ==============================================================

    plt.figure(
        figsize=(6, 6)
    )


    plt.plot(

        fpr,

        tpr,

        label=f"AUC = {roc_auc:.4f}"

    )


    plt.plot(

        [0, 1],

        [0, 1],

        linestyle="--"

    )


    plt.title(
        f"ROC Curve - {name}"
    )

    plt.xlabel(
        "FPR"
    )

    plt.ylabel(
        "TPR"
    )

    plt.legend()


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            report_dir,

            f"roc_curve_{name}.png"

        )

    )


    plt.close()


    # ==============================================================
    # PR
    # ==============================================================

    plt.figure(
        figsize=(6, 6)
    )


    plt.plot(

        rec_curve,

        prec_curve,

        label=f"AUC = {pr_auc:.4f}"

    )


    plt.title(
        f"PR Curve - {name}"
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.legend()


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            report_dir,

            f"pr_curve_{name}.png"

        )

    )


    plt.close()


    # ==============================================================
    # RETURN RESULTS
    # ==============================================================

    return {

        "train_dataset":
            "DFF_C",

        "seed":
            SEED,

        "run":
            RUN,

        "dataset":
            name,

        "accuracy":
            round(
                acc,
                4
            ),

        "precision":
            round(
                prec,
                4
            ),

        "recall":
            round(
                rec,
                4
            ),

        "f1_score":
            round(
                f1,
                4
            ),

        "roc_auc":
            round(
                roc_auc,
                4
            ),

        "pr_auc":
            round(
                pr_auc,
                4
            ),

        "eer":
            round(
                float(eer),
                4
            )
            if not np.isnan(eer)
            else "NaN",

        "total_inference_time_sec":
            round(
                inference_time,
                2
            ),

        "avg_inference_time_per_image_sec":
            round(
                avg_time,
                5
            ),

        "num_samples":
            len(dataset)

    }


# ==============================================================
# 7. MAIN
# ==============================================================

results = []


for name, path in datasets_root.items():

    if os.path.exists(path):

        metrics = evaluate_dataset(
            name,
            path
        )

        results.append(
            metrics
        )

    else:

        print(
            f"Dataset path not found: {path}"
        )


# ==============================================================
# SAVE SUMMARY RESULTS
# ==============================================================

csv_path = os.path.join(

    report_dir,

    "evaluation_results.csv"

)


json_path = os.path.join(

    report_dir,

    "evaluation_results.json"

)


pd.DataFrame(
    results
).to_csv(

    csv_path,

    index=False

)


with open(
    json_path,
    "w"
) as f:

    json.dump(

        results,

        f,

        indent=4

    )


print(
    "\nEvaluation complete."
)


print(
    f"Results saved to:\n{csv_path}"
)
