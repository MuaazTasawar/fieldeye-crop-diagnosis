"""Loads the raw ImageFolder-style dataset into a flat DataFrame.

Expects data/raw/<class_name>/*.jpg for every class in Config.class_names.
This module only *indexes* images (filepath + label) — actual image
decoding, resizing, and tensor conversion happens in preprocess.py at
train/eval time, keeping this step fast and memory-light.
"""

import os
from pathlib import Path
from typing import List

import pandas as pd

from src.config import Config

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def scan_dataset(raw_dir: str, class_names: List[str]) -> pd.DataFrame:
    """Scan an ImageFolder-style directory and index every image.

    Args:
        raw_dir: Root directory containing one subfolder per class.
        class_names: Expected class subfolder names, in a fixed order
            that also defines label indices (0..num_classes-1).

    Returns:
        DataFrame with columns ["filepath", "label", "label_idx"].

    Raises:
        FileNotFoundError: If raw_dir doesn't exist.
        ValueError: If any expected class subfolder is missing or empty,
            or if no valid images are found at all — fails loudly instead
            of silently training on an empty/partial dataset.
    """
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Raw data directory '{raw_dir}' does not exist. Create "
            f"data/raw/<class_name>/ for each of: {class_names}"
        )

    rows = []
    missing_or_empty = []
    for label_idx, class_name in enumerate(class_names):
        class_dir = root / class_name
        if not class_dir.is_dir():
            missing_or_empty.append(class_name)
            continue
        images = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in VALID_EXTENSIONS
        ]
        if not images:
            missing_or_empty.append(class_name)
            continue
        for img_path in images:
            rows.append(
                {
                    "filepath": str(img_path),
                    "label": class_name,
                    "label_idx": label_idx,
                }
            )

    if missing_or_empty:
        raise ValueError(
            "The following expected class folders are missing or empty "
            f"under '{raw_dir}': {missing_or_empty}. Every class in "
            "Config.class_names needs a matching data/raw/<class_name>/ "
            "folder with at least one image."
        )

    if not rows:
        raise ValueError(f"No valid images found under '{raw_dir}'.")

    return pd.DataFrame(rows)


def load_image_dataframe(config: Config) -> pd.DataFrame:
    """Convenience wrapper: scan the dataset using paths from Config.

    Args:
        config: Pipeline configuration.

    Returns:
        DataFrame with columns ["filepath", "label", "label_idx"].
    """
    return scan_dataset(config.raw_data_dir, config.class_names)


if __name__ == "__main__":
    cfg = Config()
    df = load_image_dataframe(cfg)
    print(f"Indexed {len(df)} images across {df['label'].nunique()} classes.")
    print(df["label"].value_counts())