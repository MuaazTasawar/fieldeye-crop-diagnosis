"""Training loop for the FieldEye crop-disease CNN.

Trains CropDiseaseCNN, checkpoints the best model by validation macro F1
(not accuracy alone — with 5 classes at ~950-1900 images each, F1 is the
more honest signal), logs every epoch to reports/runs.csv, and saves a
loss/accuracy curve figure at the end.

Usage:
    python src/train.py            # full run (Config.epochs)
    python src/train.py --quick    # fast smoke test (Config.quick_epochs,
                                    # capped train subset per class)
"""

import argparse
import csv
import os
import sys
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.getcwd())

from src.config import Config
from src.data.preprocess import get_dataloaders
from src.models.model import build_model
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with a single boolean flag: quick.
    """
    parser = argparse.ArgumentParser(description="Train the FieldEye CNN.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke-test run: fewer epochs, capped training subset per class.",
    )
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """Run one epoch of training or evaluation.

    Args:
        model: The CropDiseaseCNN.
        loader: DataLoader for this phase.
        criterion: Loss function.
        optimizer: Optimizer (only stepped if train=True).
        device: torch device.
        train: If True, run in training mode with backprop; if False,
            run in eval mode under torch.no_grad().

    Returns:
        Tuple of (avg_loss, accuracy, macro_f1) for this epoch.
    """
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, macro_f1


def plot_training_curves(history: dict, out_path: str) -> None:
    """Save loss and accuracy curves side by side.

    Args:
        history: dict with keys train_loss, val_loss, train_acc, val_acc.
        out_path: Destination PNG path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def append_run_log(row: dict, log_path: str) -> None:
    """Append one epoch's metrics to the CSV run log, writing a header
    on first write.

    Args:
        row: dict of column name -> value for this epoch.
        log_path: Destination CSV path.
    """
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """Run the full training loop and save all Phase 4 artifacts."""
    args = parse_args()
    set_seed()
    cfg = Config()
    os.makedirs(cfg.figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.checkpoint_path), exist_ok=True)

    epochs = cfg.quick_epochs if args.quick else cfg.epochs
    subset = cfg.quick_subset_per_class if args.quick else None
    print(f"Mode: {'QUICK' if args.quick else 'FULL'} | epochs={epochs} | subset_per_class={subset}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _ = get_dataloaders(cfg, subset_per_class=subset)
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_f1 = -1.0

    # Fresh run log for this run (avoid mixing runs from prior experiments).
    if os.path.isfile(cfg.runs_log_path):
        os.remove(cfg.runs_log_path)

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        elapsed = time.time() - start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        append_run_log(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_acc": round(val_acc, 4),
                "val_f1_macro": round(val_f1, 4),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            cfg.runs_log_path,
        )

        marker = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), cfg.checkpoint_path)
            marker = "  <- new best, checkpoint saved"

        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} | "
            f"val_f1_macro={val_f1:.4f} | {elapsed:.1f}s{marker}"
        )

    plot_training_curves(history, os.path.join(cfg.figures_dir, "training_curves.png"))
    print(f"\nBest val macro F1: {best_val_f1:.4f}")
    print(f"Best checkpoint saved to: {cfg.checkpoint_path}")
    print(f"Run log saved to: {cfg.runs_log_path}")
    print(f"Training curves saved to: {cfg.figures_dir}/training_curves.png")


if __name__ == "__main__":
    main()