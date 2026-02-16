"""
Evaluation utilities — inference, metrics, plots.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


CLASS_NAMES = ["Healthy", "Microcytic", "Normocytic", "Macrocytic"]


# -------------------------------------------------------------------------
# Inference
# -------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model on *loader* and collect predictions + true labels.

    Returns
    -------
    y_pred, y_true : np.ndarray of int
    """
    model.eval()
    all_preds, all_labels = [], []

    amp_enabled = device.type in ("mps", "cuda")
    use_cl = device.type in ("mps", "cuda")

    for images, cbc, labels in loader:
        if use_cl:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            images = images.to(device, non_blocking=True)
        cbc = cbc.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images, cbc)

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------

def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
):
    """Print sklearn classification report + overall accuracy."""
    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {acc:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))


# -------------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
    figsize: tuple = (7, 6),
):
    """Seaborn heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_training_curves(history: dict, figsize: tuple = (14, 5)):
    """
    Plot loss and accuracy curves.

    Parameters
    ----------
    history : dict with keys
        ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``
        (each a list of floats, one per epoch).
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
