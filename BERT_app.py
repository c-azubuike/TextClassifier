# app.py
import streamlit as st
import torch
import numpy as np
import pandas as pd
import joblib

from transformers import BertTokenizerFast, BertForSequenceClassification
from BERT_preprocessing import fast_preprocess

# Load assets
@st.cache_resource
def load_assets():
    model = BertForSequenceClassification.from_pretrained("model_assets/bert_model")
    tokenizer = BertTokenizerFast.from_pretrained("model_assets/bert_tokenizer")
    metadata = joblib.load("model_assets/bert_assets.joblib")
    label_encoder = metadata["label_encoder"]
    short_threshold = metadata["config"]["short_text_warning_length"]
    model.eval()
    return model, tokenizer, label_encoder, short_threshold

bert_model, bert_tokenizer, label_encoder, short_threshold = load_assets()

# Prediction function
def predict_with_bert(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = bert_model(**inputs).logits
    probs = torch.softmax(logits, dim=1).numpy()[0]
    pred_idx = np.argmax(probs)
    confidence = float(np.max(probs))

    preprocessed = fast_preprocess(text)
    is_short = len(preprocessed.split()) <= short_threshold

    prob_dict = {label: float(prob) * 100 for label, prob in zip(label_encoder.classes_, probs)}
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
    label = label_encoder.inverse_transform([pred_idx])[0]
    return label, confidence, is_short, sorted_probs

# Streamlit app layout
st.set_page_config(page_title="TOS Classifier", layout="centered")
st.title("TOS Content Classifier")
st.markdown("Classify fundraiser text into categories like **clear**, **adult content**, **self harm**, etc.")

text_input = st.text_area("Enter fundraiser text here:")

if "show_probs" not in st.session_state:
    st.session_state.show_probs = False

if st.button("Classify"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        label, confidence, is_short, sorted_probs = predict_with_bert(text_input)
        st.session_state.prediction_result = {
            "label": label,
            "confidence": confidence,
            "is_short": is_short,
            "sorted_probs": sorted_probs
        }
        st.session_state.show_probs = False  # Reset on new classification

if "prediction_result" in st.session_state:
    result = st.session_state.prediction_result
    st.success(f"**Prediction:** {result['label']}")
    st.info(f"**Confidence:** {result['confidence'] * 100:.2f}%")

    if result["is_short"]:
        st.warning("⚠️ This text is very short. Prediction may be less reliable.")

    if st.button("More Detail"):
        st.session_state.show_probs = not st.session_state.show_probs

    if st.session_state.show_probs:
        st.markdown("### Category Probabilities:")
        prob_df = pd.DataFrame(result["sorted_probs"].items(), columns=["Label", "Probability (%)"])
        st.table(prob_df.style.format({"Probability (%)": "{:.1f}"}))

