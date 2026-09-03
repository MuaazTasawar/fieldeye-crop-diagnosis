"""Single-image inference for the FieldEye crop-disease CNN.

Shared by both the FastAPI endpoint and the Streamlit dashboard, so
prediction logic (preprocessing, softmax, class mapping) lives in
exactly one place.
"""

import os
import sys
from typing import Dict

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.getcwd())

from src.config import Config
from src.data.preprocess import get_transforms
from src.models.model import build_model

_model = None
_device = None
_config = None


def load_model_once(config: Config = None) -> None:
    """Load the model checkpoint into a module-level cache.

    Safe to call multiple times — only loads once. Avoids reloading
    weights from disk on every single prediction request.

    Args:
        config: Pipeline configuration. Defaults to a fresh Config().

    Raises:
        FileNotFoundError: If no checkpoint exists yet.
    """
    global _model, _device, _config
    if _model is not None:
        return

    _config = config or Config()
    if not os.path.isfile(_config.checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at '{_config.checkpoint_path}'. "
            "Run `python src/train.py` first."
        )
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = build_model(_config).to(_device)
    _model.load_state_dict(torch.load(_config.checkpoint_path, map_location=_device))
    _model.eval()


def predict_image(image: Image.Image) -> Dict:
    """Run inference on a single PIL image.

    Args:
        image: A PIL Image (any mode; converted to RGB internally).

    Returns:
        dict with keys:
            "predicted_class": str, the top predicted class name.
            "confidence": float, softmax probability of the top class.
            "probabilities": dict of class_name -> probability, for
                every class (useful for a full confidence breakdown,
                not just the top-1 guess).
    """
    load_model_once()
    transform = get_transforms(train=False, image_size=_config.image_size)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    probabilities = {
        class_name: round(float(probs[i]), 4)
        for i, class_name in enumerate(_config.class_names)
    }
    top_idx = int(probs.argmax())

    return {
        "predicted_class": _config.class_names[top_idx],
        "confidence": round(float(probs[top_idx]), 4),
        "probabilities": probabilities,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("image_path", type=str, help="Path to a leaf image file.")
    args = parser.parse_args()

    img = Image.open(args.image_path)
    result = predict_image(img)
    print(f"Predicted: {result['predicted_class']} (confidence: {result['confidence']:.2%})")
    print("\nAll class probabilities:")
    for cls, prob in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {cls:<30} {prob:.2%}")