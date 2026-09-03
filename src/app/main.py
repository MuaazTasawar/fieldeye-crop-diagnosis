"""FastAPI serving layer for the FieldEye crop-disease classifier.

Run with:
    uvicorn src.app.main:app --reload --port 8000

Then POST an image to /predict, or visit /docs for interactive testing.
"""

import io
import os
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

sys.path.insert(0, os.getcwd())

from src.predict import load_model_once, predict_image

app = FastAPI(
    title="FieldEye Crop Disease Diagnosis API",
    description="Upload a tomato leaf image, get a disease diagnosis with confidence scores.",
    version="1.0.0",
)

VALID_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@app.on_event("startup")
def startup_event() -> None:
    """Load the model once when the server starts, not on first request."""
    load_model_once()


@app.get("/health")
def health_check() -> dict:
    """Simple liveness check.

    Returns:
        dict confirming the service is up.
    """
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """Diagnose a tomato leaf image.

    Args:
        file: An uploaded JPEG/PNG image of a tomato leaf.

    Returns:
        dict with predicted_class, confidence, and per-class probabilities.

    Raises:
        HTTPException(400): If the uploaded file isn't a valid image.
    """
    if file.content_type not in VALID_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{file.content_type}'. Upload a JPEG or PNG image.",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image.verify()
        image = Image.open(io.BytesIO(contents))  # re-open after verify() invalidates the handle
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image file: {exc}")

    result = predict_image(image)
    return result