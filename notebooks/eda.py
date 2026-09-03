"""Exploratory data analysis for the FieldEye crop-disease dataset.

Generates and saves (never plt.show()-only, this runs headless):
  - reports/figures/eda_class_distribution.png
  - reports/figures/eda_sample_grid.png

Run from the project root with the venv active:
    python notebooks/eda.py
"""

import os
import random
import sys

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sys.path.insert(0, os.getcwd())

from src.config import Config
from src.data.load_data import load_image_dataframe
from src.utils.seed import set_seed


def plot_class_distribution(df, out_path):
    """Save a bar chart of image counts per class.

    Args:
        df: DataFrame with a "label" column.
        out_path: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    order = df["label"].value_counts().index
    sns.countplot(y=df["label"], order=order, ax=ax)
    ax.set_title("Class Distribution — FieldEye Crop Dataset")
    ax.set_xlabel("Image Count")
    ax.set_ylabel("Class")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sample_grid(df, class_names, out_path, samples_per_class=3):
    """Save a grid of sample images, a few per class.

    Args:
        df: DataFrame with "filepath" and "label" columns.
        class_names: Ordered list of class names (defines row order).
        out_path: Destination PNG path.
        samples_per_class: Number of example images shown per class.
    """
    fig, axes = plt.subplots(
        len(class_names),
        samples_per_class,
        figsize=(samples_per_class * 2.5, len(class_names) * 2.5),
    )
    for row, class_name in enumerate(class_names):
        class_rows = df[df["label"] == class_name]
        sample_paths = class_rows["filepath"].sample(
            n=min(samples_per_class, len(class_rows)), random_state=42
        ).tolist()
        for col in range(samples_per_class):
            ax = axes[row, col] if len(class_names) > 1 else axes[col]
            ax.axis("off")
            if col < len(sample_paths):
                img = Image.open(sample_paths[col]).convert("RGB")
                ax.imshow(img)
            if col == 0:
                ax.set_title(class_name, fontsize=9, loc="left")
    fig.suptitle("Sample Images per Class", fontsize=14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    """Run the full EDA pass and print a console summary."""
    set_seed()
    cfg = Config()
    os.makedirs(cfg.figures_dir, exist_ok=True)

    df = load_image_dataframe(cfg)
    print(f"Total images: {len(df)}")
    print(df["label"].value_counts())

    plot_class_distribution(
        df, os.path.join(cfg.figures_dir, "eda_class_distribution.png")
    )
    plot_sample_grid(
        df,
        cfg.class_names,
        os.path.join(cfg.figures_dir, "eda_sample_grid.png"),
    )
    print(f"Saved figures to {cfg.figures_dir}/")


if __name__ == "__main__":
    main()