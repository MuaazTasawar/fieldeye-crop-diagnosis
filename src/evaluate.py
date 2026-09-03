"""Held-out test-set evaluation for the FieldEye CNN.

Loads the best checkpoint (by val macro F1) from training, runs it once
on the never-touched test split, and saves the diagnostics a reviewer
would actually want to see: a classification report, a confusion
matrix, and a grid of the model's actual misclassified images (not just
numbers — seeing which diseases get confused for which is the real
signal for a plant-pathology audience).

Usage:
    python src/evaluate.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

sys.path.insert(0, os.getcwd())

from src.config import Config
from src.data.preprocess import get_dataloaders
from src.models.model import build_model


def load_best_model(config: Config, device: torch.device):
    """Load the best checkpoint into a fresh model instance.

    Args:
        config: Pipeline configuration (needs checkpoint_path, num_classes).
        device: torch device to map the checkpoint onto.

    Returns:
        The model in eval() mode, weights loaded from checkpoint_path.

    Raises:
        FileNotFoundError: If no checkpoint exists yet (train.py hasn't
            been run), with a message pointing at the fix.
    """
    if not os.path.isfile(config.checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at '{config.checkpoint_path}'. "
            "Run `python src/train.py` first."
        )
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(config.checkpoint_path, map_location=device))
    model.eval()
    return model


def collect_predictions(model, loader, device):
    """Run inference over a full DataLoader and collect predictions.

    Args:
        model: Trained model in eval mode.
        loader: DataLoader to run inference over.
        device: torch device.

    Returns:
        Tuple (y_true, y_pred, filepaths) as parallel lists. filepaths
        is pulled from loader.dataset.filepaths, relying on
        shuffle=False for the test loader so order lines up.
    """
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())
    filepaths = loader.dataset.filepaths
    return y_true, y_pred, filepaths


def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    """Save a heatmap confusion matrix.

    Args:
        y_true: Ground-truth label indices.
        y_pred: Predicted label indices.
        class_names: Ordered class name list (defines axis labels).
        out_path: Destination PNG path.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Test Set")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_misclassified_grid(y_true, y_pred, filepaths, class_names, out_path, max_examples=12):
    """Save a grid of actual misclassified test images with true/pred labels.

    Args:
        y_true: Ground-truth label indices.
        y_pred: Predicted label indices.
        filepaths: Filepaths parallel to y_true/y_pred.
        class_names: Ordered class name list, for readable labels.
        out_path: Destination PNG path.
        max_examples: Max number of misclassified examples to show.
    """
    wrong_idx = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]
    if not wrong_idx:
        # Perfect test set — still save a placeholder so the pipeline
        # produces a consistent set of artifacts every run.
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No misclassified examples on test set", ha="center", va="center")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    sample = wrong_idx[:max_examples]
    n = len(sample)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for ax_i, idx in enumerate(sample):
        ax = axes[ax_i]
        img = Image.open(filepaths[idx]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"true: {class_names[y_true[idx]]}\npred: {class_names[y_pred[idx]]}",
            fontsize=8, color="red",
        )
    for ax_i in range(n, len(axes)):
        axes[ax_i].axis("off")

    fig.suptitle("Misclassified Test Examples", fontsize=14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run full held-out evaluation and save all Phase 5 artifacts."""
    cfg = Config()
    os.makedirs(cfg.figures_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_best_model(cfg, device)
    _, _, test_loader = get_dataloaders(cfg)
    y_true, y_pred, filepaths = collect_predictions(model, test_loader, device)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred, target_names=cfg.class_names, zero_division=0
    )

    print(f"Test accuracy:  {accuracy:.4f}")
    print(f"Test macro F1:  {macro_f1:.4f}")
    print("\nClassification report:\n")
    print(report)

    plot_confusion_matrix(
        y_true, y_pred, cfg.class_names,
        os.path.join(cfg.figures_dir, "confusion_matrix.png"),
    )
    plot_misclassified_grid(
        y_true, y_pred, filepaths, cfg.class_names,
        os.path.join(cfg.figures_dir, "misclassified_examples.png"),
    )

    metrics = {
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "classification_report": classification_report(
            y_true, y_pred, target_names=cfg.class_names, zero_division=0, output_dict=True
        ),
    }
    with open(cfg.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved confusion matrix to {cfg.figures_dir}/confusion_matrix.png")
    print(f"Saved misclassified examples to {cfg.figures_dir}/misclassified_examples.png")
    print(f"Saved metrics JSON to {cfg.metrics_path}")


if __name__ == "__main__":
    main()