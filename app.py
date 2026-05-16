import streamlit as st
import pandas as pd
from transformers import pipeline

st.set_page_config(page_title="Oil & Gas Incident Sentiment Analyzer", layout="wide")

st.title("🛢️ Oil & Gas Report Sentiment Analyzer")
st.subheader("Analyze technical facility logs and risk metrics using Transformers")

# Load model and cache it so it doesn't reload on every click
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="xaqren/sentiment_analysis")

pipe = load_model()

# User text input
user_input = st.text_area("Paste a technical report statement here:", height=150)

if st.button("Analyze Sentiment"):
    if user_input:
        with st.spinner("Analyzing text..."):
            result = pipe(user_input)[0]
            
            # Map the labels based on what the model returns
            raw_label = result['label']
            confidence = round(result['score'] * 100, 2)
            
            # Display results beautifully
            st.write("---")
            st.metric(label="Predicted Model Label", value=raw_label)
            st.write(f"**Confidence Level:** {confidence}%")
    else:
        st.warning("Please type or paste a report first!")