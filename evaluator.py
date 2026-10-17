# evaluator.py
import json
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)

from config import DEVICE, CHECKPOINT_DIR
from utils import banner, ok, info

# ---------------------------------------------------------------- 1. Batch Inference Helper
def collect_predictions(model, dataloader):
    """
    Runs inference on a DataLoader and returns ground-truth labels (y_true),
    predicted class indices (y_pred), and softmax probabilities (y_prob).
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.append(labels.numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    return y_true, y_pred, y_prob

# ---------------------------------------------------------------- 2. Scikit-Learn Metrics Helper
def compute_metrics(y_true, y_pred, y_prob, num_classes):
    """
    Calculates Accuracy, Top-K Accuracy, Macro Precision, Recall, F1-Score,
    and Multi-class ROC-AUC (One-vs-Rest).
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Top-K accuracy (uses k=3 for 3+ classes, or k=2 for binary)
    k = min(3, num_classes)
    top_k_acc = top_k_accuracy_score(
        y_true, y_prob, k=k, labels=np.arange(num_classes)
    )

    # Multi-class ROC-AUC
    if num_classes == 2:
        roc_auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        roc_auc = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro", labels=np.arange(num_classes)
        )

    return {
        "Accuracy": round(float(acc), 4),
        f"Top-{k} Accuracy": round(float(top_k_acc), 4),
        "Precision (Macro)": round(float(precision), 4),
        "Recall (Macro)": round(float(recall), 4),
        "F1-Score (Macro)": round(float(f1), 4),
        "ROC-AUC (OvR)": round(float(roc_auc), 4),
    }

# ---------------------------------------------------------------- 3. Per-Class Report Helper
def get_class_report_df(y_true, y_pred, class_names):
    """Generates a clean Pandas DataFrame from sklearn's classification_report."""
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    df_report = pd.DataFrame(report_dict).transpose().round(4)
    return df_report

# ---------------------------------------------------------------- 4. Master Evaluation Coordinator
def evaluate_all_models(models_dict, test_loader, class_names):
    """
    Evaluates all trained models on the test set, displays overall and per-class
    metric tables, computes confusion matrices, and saves a JSON report.
    """
    banner("SECTION 6 | Hold-Out Test Set Evaluation & Model Comparison")
    num_classes = len(class_names)
    overall_metrics = {}
    eval_artifacts = {}

    for model_name, model in models_dict.items():
        info(f"Running test set inference for {model_name}...")
        y_true, y_pred, y_prob = collect_predictions(model, test_loader)

        metrics = compute_metrics(y_true, y_pred, y_prob, num_classes)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
        df_report = get_class_report_df(y_true, y_pred, class_names)

        overall_metrics[model_name] = metrics
        eval_artifacts[model_name] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "confusion_matrix": cm,
            "classification_report": df_report,
        }

        print(f"\n--- Per-Class Classification Report: {model_name} ---")
        display(df_report)

    # Display side-by-side overall test metrics table
    print("\n--- Overall Test Set Performance Comparison ---")
    df_test_summary = pd.DataFrame.from_dict(overall_metrics, orient="index").rename_axis("Model")
    display(df_test_summary)

    # Save metrics to JSON for reproducibility records
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = CHECKPOINT_DIR / "test_metrics_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=4)
    ok(f"Saved test evaluation summary to {report_path}")

    return df_test_summary, eval_artifacts