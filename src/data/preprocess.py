"""Splits the indexed dataset and builds PyTorch Datasets/DataLoaders.

Split is stratified (preserves class proportions) and done once at the
DataFrame level, then persisted to data/processed/*.csv for
reproducibility — no images are copied or duplicated on disk, only
filepath lists. Train-time transforms include augmentation; val/test
transforms are deterministic (resize + normalize only), so evaluation
numbers reflect real generalization, not augmented luck.
"""

import os
import sys
from typing import Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.getcwd())

from src.config import Config
from src.data.load_data import load_image_dataframe

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CropDiseaseDataset(Dataset):
    """PyTorch Dataset over a DataFrame of (filepath, label_idx) rows."""

    def __init__(self, df: pd.DataFrame, transform: transforms.Compose):
        """Initialize the dataset.

        Args:
            df: DataFrame with "filepath" and "label_idx" columns.
            transform: torchvision transform pipeline applied per image.
        """
        self.filepaths = df["filepath"].tolist()
        self.labels = df["label_idx"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.filepaths[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


def get_transforms(train: bool, image_size: int) -> transforms.Compose:
    """Build the torchvision transform pipeline for train or eval.

    Args:
        train: If True, includes light augmentation (flip, rotation,
            color jitter) suited to leaf photos taken in the field
            (varying angle/lighting). If False, deterministic resize
            and normalize only, so eval numbers aren't inflated by
            augmentation-time randomness.
        image_size: Target square size images are resized to.

    Returns:
        A composed torchvision transform.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def split_dataset(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the indexed dataset and split it into train/val/test.

    Splitting is stratified on label so class proportions are preserved
    in every split, and is done in two stages (test first, then val from
    the remainder) so the requested fractions apply to the full dataset,
    not compounded fractions of a shrinking remainder.

    Args:
        config: Pipeline configuration (paths, split fractions, seed).

    Returns:
        (train_df, val_df, test_df), each with columns
        ["filepath", "label", "label_idx"].
    """
    df = load_image_dataframe(config)

    train_val_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        stratify=df["label_idx"],
        random_state=config.random_state,
    )
    val_fraction_of_remainder = config.val_size / (1 - config.test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction_of_remainder,
        stratify=train_val_df["label_idx"],
        random_state=config.random_state,
    )

    os.makedirs(config.processed_dir, exist_ok=True)
    train_df.to_csv(os.path.join(config.processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(config.processed_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(config.processed_dir, "test.csv"), index=False)

    return train_df, val_df, test_df


def _cap_per_class(df: pd.DataFrame, subset_per_class: int, random_state: int) -> pd.DataFrame:
    """Cap each class to at most `subset_per_class` rows.

    Implemented as an explicit loop + pd.concat rather than
    groupby(...).apply(...), since apply's handling of the grouping
    column is inconsistent across pandas versions and can silently drop
    "label_idx" from the result — this version is unambiguous.

    Args:
        df: DataFrame with a "label_idx" column.
        subset_per_class: Max rows to keep per label_idx value.
        random_state: Seed for reproducible sampling.

    Returns:
        A new DataFrame, capped per class, with all original columns intact.
    """
    parts = []
    for _, group in df.groupby("label_idx"):
        n = min(len(group), subset_per_class)
        parts.append(group.sample(n=n, random_state=random_state))
    return pd.concat(parts, ignore_index=True)


def get_dataloaders(
    config: Config, subset_per_class: int = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders.

    Args:
        config: Pipeline configuration.
        subset_per_class: If set, caps each class to this many training
            images (used by train.py's --quick mode for a fast smoke
            test). Val/test are left full-size so metrics stay meaningful
            even in quick mode.

    Returns:
        (train_loader, val_loader, test_loader).
    """
    train_df, val_df, test_df = split_dataset(config)

    if subset_per_class is not None:
        train_df = _cap_per_class(train_df, subset_per_class, config.random_state)

    train_ds = CropDiseaseDataset(train_df, get_transforms(train=True, image_size=config.image_size))
    val_ds = CropDiseaseDataset(val_df, get_transforms(train=False, image_size=config.image_size))
    test_ds = CropDiseaseDataset(test_df, get_transforms(train=False, image_size=config.image_size))

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    cfg = Config()
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)} ({len(train_loader.dataset)} images)")
    print(f"Val batches:   {len(val_loader)} ({len(val_loader.dataset)} images)")
    print(f"Test batches:  {len(test_loader)} ({len(test_loader.dataset)} images)")