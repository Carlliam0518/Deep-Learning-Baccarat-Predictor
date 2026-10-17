# main.py
from torchvision import datasets
from config import SEED, DEVICE, KAGGLE_DATASET, EPOCHS, LEARNING_RATE, get_env_info
from utils import info, ok, set_seed, apply_styling, run_section
from data_loader import download_dataset, build_dataloaders
from visualizer import (
    plot_class_distribution,
    show_sample_batch,
    plot_parameter_comparison,
    plot_training_curves,
    plot_confusion_matrices,
    plot_test_metric_comparison,
)
from model_builder import build_and_compare_models
from trainer import train_all_models
from evaluator import evaluate_all_models

def initialize_environment():
    """Runs Section 1: Environment setup and verification."""
    run_section(
        title="SECTION 1 | Environment and reproducibility",
        data_dict=get_env_info(),
        col_name="Value",
        index_name="Setting",
    )
    set_seed(SEED)
    apply_styling()
    info(f"Active compute device: {DEVICE}")

def prepare_data():
    """Runs Section 2: Dataset download, splitting, and DataLoader creation."""
    data_dir = download_dataset(KAGGLE_DATASET)
    dataloaders, class_names, split_indices = build_dataloaders(data_dir)

    images, labels = next(iter(dataloaders["train"]))
    ok(f"Batch sanity check -> Images shape: {tuple(images.shape)} | Labels shape: {tuple(labels.shape)}")
    return data_dir, dataloaders, class_names, split_indices

def visualize_data(data_dir, dataloaders, class_names, split_indices):
    """Runs Section 3: Class balance verification and sample image preview."""
    raw_ds = datasets.ImageFolder(root=data_dir)
    plot_class_distribution(raw_ds.targets, split_indices, class_names)
    show_sample_batch(dataloaders["train"], class_names, split_name="train", num_images=8, cols=4)

def setup_models(class_names):
    """Runs Section 4: Initializes ResNet50 and MobileNetV3 and compares parameters."""
    models_dict, df_params = build_and_compare_models(
        num_classes=len(class_names),
        model_names=("ResNet50", "MobileNetV3"),
        freeze_backbone=True,
    )
    plot_parameter_comparison(df_params)
    return models_dict, df_params

def run_training(models_dict, dataloaders):
    """Runs Section 5: Trains all models, saves best weights, and plots learning curves."""
    models_dict, histories, df_train_summary = train_all_models(
        models_dict,
        dataloaders,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
    )
    plot_training_curves(histories)
    return models_dict, histories, df_train_summary

def run_evaluation(models_dict, dataloaders, class_names):
    """Runs Section 6: Evaluates trained models on the test set and plots results."""
    df_test_summary, eval_artifacts = evaluate_all_models(
        models_dict,
        dataloaders["test"],
        class_names,
    )
    plot_confusion_matrices(eval_artifacts, class_names)
    plot_test_metric_comparison(df_test_summary)
    return df_test_summary, eval_artifacts

if __name__ == "__main__":
    initialize_environment()
    data_dir, dataloaders, class_names, split_indices = prepare_data()
    visualize_data(data_dir, dataloaders, class_names, split_indices)
    models_dict, df_params = setup_models(class_names)
    models_dict, histories, df_train_summary = run_training(models_dict, dataloaders)
    df_test_summary, eval_artifacts = run_evaluation(models_dict, dataloaders, class_names)