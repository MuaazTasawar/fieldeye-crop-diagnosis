"""Streamlit demo for the FieldEye crop-disease classifier.

Run with:
    streamlit run src/app/dashboard.py

Upload a tomato leaf photo and get an instant diagnosis with a
confidence breakdown across all 5 classes.
"""

import os
import sys

import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, os.getcwd())

from src.predict import predict_image

st.set_page_config(page_title="FieldEye — Crop Disease Diagnosis", page_icon="🌿")

st.title("🌿 FieldEye — Tomato Leaf Disease Diagnosis")
st.caption(
    "Upload a photo of a tomato leaf to get an instant diagnosis. "
    "Trained on 5 classes: healthy, early blight, late blight, leaf mold, septoria leaf spot."
)

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Diagnosing..."):
        result = predict_image(image)

    with col2:
        predicted = result["predicted_class"].replace("_", " ").title()
        confidence = result["confidence"]

        if result["predicted_class"] == "tomato_healthy":
            st.success(f"**{predicted}** — {confidence:.1%} confidence")
        else:
            st.warning(f"**{predicted}** — {confidence:.1%} confidence")

        st.subheader("Confidence breakdown")
        probs_df = pd.DataFrame(
            {
                "Class": [c.replace("_", " ").title() for c in result["probabilities"].keys()],
                "Probability": list(result["probabilities"].values()),
            }
        ).sort_values("Probability", ascending=True)
        st.bar_chart(probs_df.set_index("Class"))

st.divider()
st.caption(
    "⚠️ This is a portfolio/research project, not a substitute for professional "
    "agricultural diagnosis. Model trained on the PlantVillage dataset."
)