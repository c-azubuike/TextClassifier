# app.py
import streamlit as st
import joblib
from preprocessing import fast_preprocess

# Load model
assets = joblib.load("model_assets.pkl")
model = assets["model"]
label_encoder = assets["label_encoder"]
short_threshold = assets["config"]["short_text_warning_length"]

st.set_page_config(page_title="Text Classifier", layout="centered")
st.title(" TOS Content Category Classifier")
st.markdown("Enter fundraiser description and classify it into categories (e.g. adult content, self harm, crypto promotion, etc.)")

text_input = st.text_area("Input Text")

if st.button("Classify"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        preprocessed = fast_preprocess(text_input)
        word_count = len(preprocessed.split())
        pred = model.predict([preprocessed])[0]
        proba = max(model.predict_proba([preprocessed])[0])
        label = label_encoder.inverse_transform([pred])[0]

        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: **{proba * 100:.2f}%**")

        if word_count <= short_threshold:
            st.warning("⚠️ Input is short — prediction may be less reliable.")
