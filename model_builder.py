# model_builder.py
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from IPython.display import display

from config import DEVICE, SEED
from utils import banner, ok, info, set_seed

# ---------------------------------------------------------------- 1. Parameter & Freezing Helpers
def set_backbone_trainable(model, trainable=False):
    """Freezes or unfreezes all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = trainable

def count_parameters(model):
    """
    Computes total, trainable, and frozen parameter counts for any PyTorch model.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    trainable_pct = (trainable / total) * 100 if total > 0 else 0.0

    return {
        "Total Params": total,
        "Trainable Params": trainable,
        "Frozen Params": frozen,
        "Trainable (%)": round(trainable_pct, 2),
    }

# ---------------------------------------------------------------- 2. Unified Architecture Builder
def build_model(model_name, num_classes, freeze_backbone=True, dropout_p=0.3):
    """
    Loads a pretrained backbone ('ResNet50' or 'MobileNetV3'), optionally freezes
    feature extractor weights, replaces the classifier head, and sends to DEVICE.
    """
    set_seed(SEED)

    if model_name == "ResNet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        if freeze_backbone:
            set_backbone_trainable(model, trainable=False)

        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "MobileNetV3":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
        if freeze_backbone:
            set_backbone_trainable(model, trainable=False)

        # Replace the final linear layer inside MobileNetV3's classifier block
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        # Ensure the entire classifier head is trainable
        for param in model.classifier.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unsupported model_name: '{model_name}'. Choose 'ResNet50' or 'MobileNetV3'.")

    model = model.to(DEVICE)
    stats = count_parameters(model)
    ok(
        f"Initialized {model_name} on {DEVICE} | "
        f"Trainable: {stats['Trainable Params']:,} / {stats['Total Params']:,} ({stats['Trainable (%)']}%)"
    )
    return model

# ---------------------------------------------------------------- 3. Multi-Model Comparison Helper
def build_and_compare_models(num_classes, model_names=("ResNet50", "MobileNetV3"), freeze_backbone=True):
    """
    Builds a dictionary of models and displays a side-by-side parameter comparison table.
    """
    banner("SECTION 4 | Model Architecture Setup & Parameter Comparison")
    models_dict = {}
    summary_rows = {}

    for name in model_names:
        model = build_model(name, num_classes=num_classes, freeze_backbone=freeze_backbone)
        models_dict[name] = model
        summary_rows[name] = count_parameters(model)

    df_params = pd.DataFrame.from_dict(summary_rows, orient="index").rename_axis("Model")
    display(df_params)
    return models_dict, df_params