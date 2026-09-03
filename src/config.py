"""Central configuration for the FieldEye crop-disease classifier.

No hyperparameters or paths live anywhere else in the codebase — every
script imports its settings from here, so a single edit changes behavior
consistently across data loading, training, evaluation, and serving.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Pipeline-wide configuration.

    Attributes:
        raw_data_dir: Root of the ImageFolder-style raw dataset
            (data/raw/<class_name>/*.jpg).
        processed_dir: Where train/val/test splits are written after
            preprocessing.
        class_names: The disease/healthy classes used for this MVP scope.
            Fixed list (not auto-discovered) so class order stays stable
            across training runs and served predictions.
        image_size: Height/width (square) images are resized to before
            the model sees them.
        batch_size: Training/eval batch size.
        num_workers: DataLoader worker processes.
        val_size: Fraction of data held out for validation.
        test_size: Fraction of data held out for the final test set.
        epochs: Full-run epoch count.
        quick_epochs: Epoch count for the --quick smoke-test config.
        quick_subset_per_class: Max images per class used in --quick mode.
        learning_rate: Adam optimizer learning rate.
        weight_decay: Adam optimizer weight decay (L2 regularization).
        random_state: Seed used for all splits and shuffling.
        checkpoint_path: Where the best model (by val macro F1) is saved.
        figures_dir: Where all matplotlib/seaborn PNGs are saved.
        runs_log_path: CSV log of per-epoch training metrics.
        metrics_path: JSON file with final held-out test metrics.
    """

    raw_data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    class_names: List[str] = field(
        default_factory=lambda: [
            "tomato_healthy",
            "tomato_early_blight",
            "tomato_late_blight",
            "tomato_leaf_mold",
            "tomato_septoria_leaf_spot",
        ]
    )
    image_size: int = 128
    batch_size: int = 32
    num_workers: int = 2
    val_size: float = 0.15
    test_size: float = 0.15
    epochs: int = 20
    quick_epochs: int = 2
    quick_subset_per_class: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    random_state: int = 42
    checkpoint_path: str = "reports/best_model.pt"
    figures_dir: str = "reports/figures"
    runs_log_path: str = "reports/runs.csv"
    metrics_path: str = "reports/metrics.json"

    @property
    def num_classes(self) -> int:
        """Number of target classes, derived from class_names."""
        return len(self.class_names)