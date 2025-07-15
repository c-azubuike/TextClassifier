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
    label = label_encoder.inverse_transform([pred_idx])[0]

    # Return full probability dictionary with percentages
    prob_dict = {label: float(prob) * 100 for label, prob in zip(label_encoder.classes_, probs)}
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))

    return label, confidence, is_short, sorted_probs

# App UI
st.set_page_config(page_title="TOS Classifier", layout="centered")
st.title("TOS Fundraiser Classifier")
st.markdown("Classify fundraiser text into categories like **clear**, **adult content**, **self harm**, etc.")

text_input = st.text_area("Enter fundraiser text here:")

if st.button("Classify"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        label, confidence, is_short, sorted_probs = predict_with_bert(text_input)

        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence * 100:.2f}%")

        if is_short:
            st.warning("⚠️ This text is very short. Prediction may be less reliable.")

        # More Detail button for full probabilities
        if st.button("More Detail"):
            st.markdown("### Category Probabilities:")
            prob_df = pd.DataFrame(sorted_probs.items(), columns=["Label", "Probability (%)"])
            st.table(prob_df.style.format({"Probability (%)": "{:.1f}"}))
