# config.py
import os
import json
import platform
from pathlib import Path
import torch
import torchvision

# 1. Reproducibility & Hardware Constants
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Dataset & DataLoader Constants (Playing Cards Dataset)
KAGGLE_DATASET = "gpiosenka/cards-image-datasetclassification"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = min(4, os.cpu_count() or 1)

# Split Ratios (Must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Standard ImageNet Normalization (Required for ResNet50 / MobileNetV3)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 3. Training Hyperparameters & Checkpoints
EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
CHECKPOINT_DIR = Path("checkpoints")
CLASS_NAMES_PATH = CHECKPOINT_DIR / "class_names.json"
ROIS_JSON_PATH = CHECKPOINT_DIR / "card_rois.json"

# 4. Live Screen-Capture Settings
CONFIDENCE_THRESHOLD = 0.65  # Ignore empty table felt if confidence is below 65%
SCAN_INTERVAL_SEC = 1.0      # Time between screen scans in seconds

# Default fallback bounding boxes if calibrate_rois.py hasn't been run yet
DEFAULT_CARD_ROIS = {
    "Player_Card_1": {"top": 420, "left": 380, "width": 110, "height": 155},
    "Player_Card_2": {"top": 420, "left": 500, "width": 110, "height": 155},
    "Player_Card_3": {"top": 420, "left": 620, "width": 110, "height": 155},
    "Banker_Card_1": {"top": 420, "left": 820, "width": 110, "height": 155},
    "Banker_Card_2": {"top": 420, "left": 940, "width": 110, "height": 155},
    "Banker_Card_3": {"top": 420, "left": 1060, "width": 110, "height": 155},
}

def load_card_rois():
    """
    Loads custom calibrated screen coordinates from checkpoints/card_rois.json
    if available; otherwise falls back to DEFAULT_CARD_ROIS.
    """
    if ROIS_JSON_PATH.exists():
        with open(ROIS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CARD_ROIS

CARD_ROIS = load_card_rois()

# 5. Shared Color Palette for Plots
PALETTE = {
    "train": "#4C78A8",
    "validation": "#F58518",
    "test": "#54A24B",
    "trainable": "#2A9D8F",
    "frozen": "#B8BEC6",
    "ResNet50": "#7F3C8D",
    "MobileNetV3": "#11A579",
}

# 6. Matplotlib Styling Dictionary
PLOT_STYLE = {
    "figure.dpi": 110,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.frameon": False,
    "font.size": 10,
}

def get_env_info():
    """Returns a dictionary of current system and library specifications."""
    return {
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "none (CPU)",
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "python": platform.python_version(),
        "seed": SEED,
    }