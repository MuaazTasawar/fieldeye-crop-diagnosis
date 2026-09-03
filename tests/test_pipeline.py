"""Lightweight tests for the FieldEye pipeline.

These check the pieces that don't require a full trained checkpoint or
populated dataset (config shape, model forward pass, transform output
shape) so they run anywhere, plus a couple of integration tests that
skip gracefully if data/raw or the checkpoint aren't present — useful
for CI environments that won't have the ~680MB PlantVillage subset.

Run with:
    pytest tests/ -v
"""

import os
import sys

import pytest
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())

from src.config import Config
from src.data.preprocess import get_transforms
from src.models.model import build_model, CropDiseaseCNN


@pytest.fixture
def config():
    """A fresh default Config for each test."""
    return Config()


def test_config_num_classes_matches_class_names(config):
    """num_classes property should always match len(class_names)."""
    assert config.num_classes == len(config.class_names) == 5


def test_model_forward_pass_shape(config):
    """Model output shape should be (batch, num_classes) for any batch size."""
    model = build_model(config)
    dummy = torch.randn(4, 3, config.image_size, config.image_size)
    output = model(dummy)
    assert output.shape == (4, config.num_classes)


def test_model_is_correct_type(config):
    """build_model should return a CropDiseaseCNN instance."""
    model = build_model(config)
    assert isinstance(model, CropDiseaseCNN)


def test_transforms_produce_correct_tensor_shape(config):
    """Both train and eval transforms should resize to (3, image_size, image_size)."""
    dummy_image = Image.new("RGB", (300, 200), color="green")
    for train_flag in (True, False):
        transform = get_transforms(train=train_flag, image_size=config.image_size)
        tensor = transform(dummy_image)
        assert tensor.shape == (3, config.image_size, config.image_size)


def test_transforms_are_normalized(config):
    """Eval transform output should not be in raw [0, 255] pixel range."""
    dummy_image = Image.new("RGB", (300, 200), color="white")
    transform = get_transforms(train=False, image_size=config.image_size)
    tensor = transform(dummy_image)
    # After ImageNet normalization, a solid white image should NOT sit at ~1.0
    # (ToTensor alone would put white at 1.0; normalization shifts it away).
    assert tensor.max().item() != pytest.approx(1.0, abs=1e-3)


@pytest.mark.skipif(
    not os.path.isdir("data/raw"), reason="data/raw not populated in this environment"
)
def test_dataset_loads_from_raw_directory(config):
    """If data/raw exists, load_image_dataframe should index at least one image per class."""
    from src.data.load_data import load_image_dataframe

    df = load_image_dataframe(config)
    assert len(df) > 0
    assert set(df["label"].unique()) == set(config.class_names)


@pytest.mark.skipif(
    not os.path.isfile("reports/best_model.pt"),
    reason="No trained checkpoint present — run train.py first",
)
def test_predict_image_returns_valid_output(config):
    """If a checkpoint exists, predict_image should return a well-formed result dict."""
    from src.predict import predict_image

    dummy_image = Image.new("RGB", (256, 256), color="green")
    result = predict_image(dummy_image)

    assert result["predicted_class"] in config.class_names
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probabilities"].keys()) == set(config.class_names)
    assert abs(sum(result["probabilities"].values()) - 1.0) < 0.01