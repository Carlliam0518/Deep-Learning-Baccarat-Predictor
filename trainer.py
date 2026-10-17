# trainer.py
import json
import time
import copy
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import display

from config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CHECKPOINT_DIR,
    CLASS_NAMES_PATH,
)
from utils import banner, ok, info

# ---------------------------------------------------------------- 1. Optimizer & Loss Setup Helper
def configure_training(model, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, epochs=EPOCHS):
    """
    Sets up CrossEntropyLoss, AdamW optimizer (only on trainable parameters),
    and a CosineAnnealing learning rate scheduler.
    """
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return criterion, optimizer, scheduler

# ---------------------------------------------------------------- 2. Unified Single-Epoch Helper
def run_epoch(model, dataloader, criterion, optimizer=None, is_train=True):
    """
    Executes one full pass over a DataLoader.
    Works for both Training (is_train=True) and Validation/Test (is_train=False).
    """
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in dataloader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct_preds += (preds == labels).sum().item()
            total_samples += batch_size

    epoch_loss = running_loss / max(total_samples, 1)
    epoch_acc = correct_preds / max(total_samples, 1)
    return epoch_loss, epoch_acc

# ---------------------------------------------------------------- 3. Complete Model Training Loop Helper
def train_model(model_name, model, dataloaders, epochs=EPOCHS, lr=LEARNING_RATE):
    """
    Trains and validates a model across multiple epochs, tracks metrics,
    restores the best validation weights, and saves a .pth checkpoint.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    criterion, optimizer, scheduler = configure_training(model, lr=lr, epochs=epochs)

    # Save class names from the underlying dataset if not already saved
    underlying_ds = dataloaders["train"].dataset.dataset
    if hasattr(underlying_ds, "classes") and not CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
            json.dump(underlying_ds.classes, f, indent=4)
        ok(f"Saved {len(underlying_ds.classes)} card class names to {CLASS_NAMES_PATH}")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "epoch_time": [],
    }

    best_val_acc = -1.0
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    ckpt_path = CHECKPOINT_DIR / f"{model_name}_best.pth"

    info(f"Starting training for {model_name} ({epochs} epochs, lr={lr})...")
    start_total = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, dataloaders["train"], criterion, optimizer=optimizer, is_train=True
        )
        val_loss, val_acc = run_epoch(
            model, dataloaders["validation"], criterion, optimizer=None, is_train=False
        )
        scheduler.step()

        elapsed = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(elapsed)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, ckpt_path)

        tag = " *BEST*" if is_best else ""
        print(
            f"    Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s{tag}"
        )

    total_time = time.time() - start_total
    model.load_state_dict(best_weights)
    ok(
        f"{model_name} complete in {total_time:.1f}s | "
        f"Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch}) -> Saved to {ckpt_path}"
    )

    summary = {
        "Best Epoch": best_epoch,
        "Best Val Acc": round(best_val_acc, 4),
        "Final Train Acc": round(history["train_acc"][-1], 4),
        "Min Val Loss": round(min(history["val_loss"]), 4),
        "Total Time (s)": round(total_time, 1),
        "Checkpoint": str(ckpt_path),
    }
    return model, history, summary

# ---------------------------------------------------------------- 4. Multi-Model Training Coordinator
def train_all_models(models_dict, dataloaders, epochs=EPOCHS, lr=LEARNING_RATE):
    """
    Loops through all initialized models, trains each one, and displays
    a side-by-side training summary table.
    """
    banner("SECTION 5 | Model Training & Validation Checkpointing")
    histories = {}
    summaries = {}

    for model_name, model in models_dict.items():
        trained_model, history, summary = train_model(
            model_name, model, dataloaders, epochs=epochs, lr=lr
        )
        models_dict[model_name] = trained_model
        histories[model_name] = history
        summaries[model_name] = summary

    df_train_summary = pd.DataFrame.from_dict(summaries, orient="index").rename_axis("Model")
    display(df_train_summary)
    return models_dict, histories, df_train_summary