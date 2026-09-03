"""Compact custom CNN for the FieldEye crop-disease classifier.

Kept small on purpose (4 conv blocks, no pretrained backbone) so a full
training run stays feasible on CPU. Trade-off: a fine-tuned
torchvision.models.resnet18 backbone would likely reach higher accuracy
faster with less data, at the cost of a much larger model and requiring
GPU for reasonable training time — noted as the upgrade path in the
README rather than built here.
"""

import os
import sys
from typing import List

import torch
import torch.nn as nn

sys.path.insert(0, os.getcwd())

from src.config import Config


class CropDiseaseCNN(nn.Module):
    """4-block convolutional classifier for square RGB leaf images."""

    def __init__(self, num_classes: int, in_channels: int = 3):
        """Build the network.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input image channels (3 for RGB).
        """
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(in_channels, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        """One Conv -> BatchNorm -> ReLU -> MaxPool block.

        Args:
            in_ch: Input channel count.
            out_ch: Output channel count.

        Returns:
            The block as an nn.Sequential.
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input batch, shape (B, in_channels, H, W).

        Returns:
            Raw class logits, shape (B, num_classes).
        """
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(config: Config) -> CropDiseaseCNN:
    """Construct the model from pipeline config.

    Args:
        config: Pipeline configuration (needs num_classes).

    Returns:
        An initialized CropDiseaseCNN.
    """
    return CropDiseaseCNN(num_classes=config.num_classes)


def architecture_summary(config: Config) -> List[str]:
    """Human-readable layer list for the architecture diagram.

    Args:
        config: Pipeline configuration (needs image_size, num_classes).

    Returns:
        Ordered list of layer descriptions, input to output.
    """
    s = config.image_size
    sizes = [s // (2 ** i) for i in range(5)]  # after each of 4 pools
    return [
        f"Input 3x{sizes[0]}x{sizes[0]}",
        f"Conv 32 + Pool -> {sizes[1]}x{sizes[1]}",
        f"Conv 64 + Pool -> {sizes[2]}x{sizes[2]}",
        f"Conv 128 + Pool -> {sizes[3]}x{sizes[3]}",
        f"Conv 256 + Pool -> {sizes[4]}x{sizes[4]}",
        "AdaptiveAvgPool -> 256",
        "FC 128 + Dropout",
        f"FC {config.num_classes} (logits)",
    ]


def plot_architecture_diagram(layers: List[str], out_path: str) -> None:
    """Save a simple box-and-arrow diagram of the architecture.

    Args:
        layers: Ordered list of layer description strings.
        out_path: Destination PNG path.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, len(layers) * 0.8))
    for i, label in enumerate(layers):
        y = len(layers) - i
        ax.add_patch(
            plt.Rectangle((0.2, y - 0.4), 2.4, 0.6, fill=True, facecolor="#4C72B0", edgecolor="black")
        )
        ax.text(1.4, y - 0.1, label, ha="center", va="center", color="white", fontsize=8)
        if i > 0:
            ax.annotate(
                "", xy=(1.4, y + 0.2), xytext=(1.4, y + 0.6), arrowprops=dict(arrowstyle="->")
            )
    ax.set_xlim(0, 2.8)
    ax.set_ylim(0, len(layers) + 1)
    ax.axis("off")
    ax.set_title("CropDiseaseCNN Architecture")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    cfg = Config()
    model = build_model(cfg)
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Sanity check: forward pass shape
    dummy_input = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    output = model(dummy_input)
    print(f"Forward pass check: input {tuple(dummy_input.shape)} -> output {tuple(output.shape)}")

    os.makedirs(cfg.figures_dir, exist_ok=True)
    plot_architecture_diagram(
        architecture_summary(cfg), os.path.join(cfg.figures_dir, "model_architecture.png")
    )
    print(f"Saved architecture diagram to {cfg.figures_dir}/model_architecture.png")