# data_loader.py
from pathlib import Path
import numpy as np
import kagglehub
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from config import (
    SEED,
    DEVICE,
    KAGGLE_DATASET,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from utils import ok, info, warn, run_section

# ---------------------------------------------------------------- 1. Download & Path Discovery Helpers
def find_image_root(base_path):
    """
    Walks through the downloaded directory to automatically find the folder
    containing class subdirectories (compatible with torchvision ImageFolder).
    """
    base = Path(base_path)
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for candidate in [base] + [p for p in base.rglob("*") if p.is_dir()]:
        subdirs = [d for d in candidate.iterdir() if d.is_dir()]
        if len(subdirs) >= 2:
            # Check if subdirectories actually contain image files
            has_images = any(
                f.suffix.lower() in IMAGE_EXTS
                for sub in subdirs[:3]
                for f in sub.iterdir()
                if f.is_file()
            )
            if has_images:
                return candidate
    raise FileNotFoundError(f"Could not find valid class folders inside {base_path}")

def download_dataset(dataset_handle=KAGGLE_DATASET):
    """Downloads a Kaggle dataset and returns the resolved path to the image folders."""
    info(f"Fetching dataset from Kaggle: '{dataset_handle}'...")
    raw_path = kagglehub.dataset_download(dataset_handle)
    data_dir = find_image_root(raw_path)
    ok(f"Dataset located at: {data_dir}")
    return data_dir

# ---------------------------------------------------------------- 2. Image Transform Helper
def get_transforms(img_size=IMG_SIZE):
    """
    Returns a dictionary of transforms for 'train' (with augmentation)
    and 'eval' (deterministic center crop for validation/test).
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            normalize,
        ]),
        "eval": transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]),
    }

# ---------------------------------------------------------------- 3. Stratified Splitting Helper
def create_stratified_splits(targets, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=SEED):
    """
    Splits dataset indices into stratified train, validation, and test sets.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    indices = np.arange(len(targets))
    targets = np.array(targets)

    # First split: Train vs Temp (Validation + Test)
    train_idx, temp_idx, _, temp_targets = train_test_split(
        indices,
        targets,
         train_size=train_ratio,
        stratify=targets,
        random_state=seed,
    )

    # Second split: Divide Temp into Validation and Test
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=relative_val_ratio,
        stratify=temp_targets,
        random_state=seed,
    )

    ok(f"Stratified split complete -> Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    return {"train": train_idx, "validation": val_idx, "test": test_idx}

# ---------------------------------------------------------------- 4. DataLoader Builder Helper
def build_dataloaders(data_dir, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Master helper function: builds datasets, applies split-specific transforms,
    creates DataLoaders, and prints a summary table.
    """
    tfms = get_transforms()

    # Create two dataset instances so train augmentations don't affect val/test
    train_full_ds = datasets.ImageFolder(root=data_dir, transform=tfms["train"])
    eval_full_ds = datasets.ImageFolder(root=data_dir, transform=tfms["eval"])

    class_names = train_full_ds.classes
    split_indices = create_stratified_splits(train_full_ds.targets)

    # Map each split to its proper underlying dataset and shuffle rule
    subsets = {
        "train": Subset(train_full_ds, split_indices["train"]),
        "validation": Subset(eval_full_ds, split_indices["validation"]),
        "test": Subset(eval_full_ds, split_indices["test"]),
    }

    pin_memory = DEVICE.type == "cuda"
    dataloaders = {}

    for split_name, subset in subsets.items():
        is_train = split_name == "train"
        dataloaders[split_name] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=is_train,  # Keeps batch sizes uniform during training
        )

    # Print clean summary using our utils.py helper
    summary_stats = {
        "Dataset Source": KAGGLE_DATASET,
        "Resolved Directory": str(data_dir),
        "Total Images": len(train_full_ds),
        "Number of Classes": len(class_names),
        "Train Samples (Batches)": f"{len(subsets['train'])} ({len(dataloaders['train'])})",
        "Validation Samples (Batches)": f"{len(subsets['validation'])} ({len(dataloaders['validation'])})",
        "Test Samples (Batches)": f"{len(subsets['test'])} ({len(dataloaders['test'])})",
        "Batch Size": batch_size,
        "Image Resolution": f"{IMG_SIZE}x{IMG_SIZE}",
    }
    run_section("SECTION 2 | Dataset Preparation & DataLoaders", summary_stats, col_name="Details", index_name="Metric")

    return dataloaders, class_names, split_indices