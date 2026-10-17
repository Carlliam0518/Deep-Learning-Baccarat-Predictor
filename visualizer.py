# visualizer.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from config import PALETTE, IMAGENET_MEAN, IMAGENET_STD
from utils import banner, ok

# ---------------------------------------------------------------- 1. Image & Plot Helpers
def denormalize_image(tensor):
    """
    Converts a normalized PyTorch image tensor (C, H, W) back into a
    standard NumPy array (H, W, C) clipped between 0 and 1 for plotting.
    """
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = std * img + mean
    return np.clip(img, 0.0, 1.0)

def finalize_plot(suptitle=None):
    """Applies consistent spacing, optional super-title, and renders the figure."""
    if suptitle:
        plt.suptitle(suptitle, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------- 2. Class Balance Verification
def plot_class_distribution(targets, split_indices, class_names):
    """
    Computes and plots the number of images per class across Train,
    Validation, and Test splits to verify stratification.
    """
    banner("SECTION 3 | Exploratory Data Analysis & Class Distribution")
    targets = np.array(targets)

    counts = {}
    for split_name, indices in split_indices.items():
        split_targets = targets[indices]
        counts[split_name.capitalize()] = [
            int(np.sum(split_targets == cls_idx)) for cls_idx in range(len(class_names))
        ]

    df_counts = pd.DataFrame(counts, index=class_names).rename_axis("Class Name")
    df_counts["Total"] = df_counts.sum(axis=1)
    display(df_counts)

    ax = df_counts[["Train", "Validation", "Test"]].plot(
        kind="bar",
        figsize=(11, 4.5),
        color=[PALETTE["train"], PALETTE["validation"], PALETTE["test"]],
        width=0.8,
    )
    ax.set_title("Class Distribution Across Data Partitions")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    plt.xticks(rotation=30, ha="right")
    finalize_plot()
    ok("Class distribution table and bar chart generated")
    return df_counts

# ---------------------------------------------------------------- 3. Sample Batch Grid Helper
def show_sample_batch(dataloader, class_names, split_name="train", num_images=8, cols=4):
    """
    Fetches a single batch from any DataLoader, reverses normalization,
    and plots a clean grid of sample images with their class labels.
    """
    images, labels = next(iter(dataloader))
    num_images = min(num_images, len(images))
    rows = int(np.ceil(num_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.8))
    axes = np.array(axes).reshape(-1)

    for idx in range(len(axes)):
        axes[idx].axis("off")
        if idx < num_images:
            img_np = denormalize_image(images[idx])
            label_name = class_names[labels[idx].item()]
            axes[idx].imshow(img_np)
            axes[idx].set_title(label_name, fontsize=9, color=PALETTE.get(split_name, "#333333"))

    finalize_plot(suptitle=f"Sample Images from '{split_name.upper()}' Partition")
    ok(f"Displayed {num_images} sample images from the {split_name} DataLoader")

# ---------------------------------------------------------------- 4. Model Parameter Comparison Plot
def plot_parameter_comparison(df_params):
    """
    Plots a horizontal bar chart comparing Trainable vs. Frozen parameters
    across initialized models using the shared PALETTE.
    """
    plot_df = df_params[["Trainable Params", "Frozen Params"]] / 1e6  # Convert to Millions
    ax = plot_df.plot(
        kind="barh",
        stacked=True,
        figsize=(8, 3.5),
        color=[PALETTE["trainable"], PALETTE["frozen"]],
        width=0.6,
    )
    ax.set_title("Model Complexity: Trainable vs. Frozen Parameters (Millions)")
    ax.set_xlabel("Parameters (Millions)")
    ax.set_ylabel("Architecture")
    finalize_plot()
    ok("Model parameter comparison chart generated")

# ---------------------------------------------------------------- 5. Training & Validation Learning Curves
def plot_training_curves(histories):
    """
    Plots side-by-side Loss and Accuracy learning curves comparing all models
    across epochs using solid lines for Validation and dashed lines for Train.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for model_name, hist in histories.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        color = PALETTE.get(model_name, "#333333")

        axes[0].plot(epochs, hist["train_loss"], linestyle="--", alpha=0.65, color=color, label=f"{model_name} (Train)")
        axes[0].plot(epochs, hist["val_loss"], linestyle="-", linewidth=2, color=color, label=f"{model_name} (Val)")

        axes[1].plot(epochs, hist["train_acc"], linestyle="--", alpha=0.65, color=color, label=f"{model_name} (Train)")
        axes[1].plot(epochs, hist["val_acc"], linestyle="-", linewidth=2, color=color, label=f"{model_name} (Val)")

    axes[0].set_title("Cross-Entropy Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].set_title("Classification Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    finalize_plot(suptitle="Training vs. Validation Learning Curves")
    ok("Learning curve plots generated")

# ---------------------------------------------------------------- 6. Confusion Matrices & Test Metric Plots
def plot_confusion_matrices(eval_artifacts, class_names):
    """
    Plots side-by-side annotated Confusion Matrices for all evaluated models.
    """
    num_models = len(eval_artifacts)
    fig, axes = plt.subplots(1, num_models, figsize=(6.2 * num_models, 5.5))
    if num_models == 1:
        axes = [axes]

    for ax, (model_name, data) in zip(axes, eval_artifacts.items()):
        cm = data["confusion_matrix"]
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{model_name} | Confusion Matrix")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.grid(False)

        # Annotate cell counts
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    finalize_plot(suptitle="Test Set Confusion Matrices")
    ok("Confusion matrix plots generated")

def plot_test_metric_comparison(df_test_summary):
    """
    Plots a grouped bar chart comparing ResNet50 and MobileNetV3 across all
    computed test set evaluation metrics.
    """
    df_transposed = df_test_summary.transpose()
    colors = [PALETTE.get(col, "#555555") for col in df_transposed.columns]

    ax = df_transposed.plot(kind="bar", figsize=(10, 4.5), color=colors, width=0.75)
    ax.set_title("Test Set Metrics: ResNet50 vs. MobileNetV3")
    ax.set_xlabel("Evaluation Metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.08)
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Model")
    finalize_plot()
    ok("Test metric comparison bar chart generated")