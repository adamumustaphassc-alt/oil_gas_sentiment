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
            
            raw_label = result['label']
            confidence = round(result['score'] * 100, 2)
            
            # 1. Define human-readable mapping
            label_mapping = {
                "LABEL_0": "Clear / Normal Operations",
                "LABEL_1": "Minor Warning / Maintenance Needed",
                "LABEL_2": "Critical Risk / Facility Emergency"
            }
            
            # 2. Translate the raw label safely
            clean_label = label_mapping.get(raw_label, f"Unknown Code ({raw_label})")
            
            # Display results beautifully
            st.write("---")
            
            # Create a green visual for clear ops, or red for risk
            if "Clear" in clean_label:
                st.success(f"🟢 **Status:** {clean_label}")
            else:
                st.error(f"🚨 **Status:** {clean_label}")
                
            st.metric(label="Model Confidence Level", value=f"{confidence}%")
            st.info(f"Raw Internal Model Output: `{raw_label}`")
    else:
        st.warning("Please type or paste a report first!")