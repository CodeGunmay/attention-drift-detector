"""
Attention Drift Detection System
Author: Gunmay Parganiha
Graphic Era University, Dehradun

Streamlit app that predicts attention state from typing patterns.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
import joblib

st.set_page_config(page_title="Attention Drift Detector", layout="wide")

st.title("Attention Drift Detection System")
st.markdown("*Detects when you lose focus while typing*")

@st.cache_resource
def load_model():
    try:
        if os.path.exists('distraction_detector_model.pkl'):
            model = joblib.load('distraction_detector_model.pkl')
            return model, True
        return None, False
    except Exception as e:
        st.error(f"Error: {e}")
        return None, False

model, model_loaded = load_model()

if model_loaded:
    st.success("Model loaded successfully")
    
    with st.sidebar:
        st.header("Model Info")
        st.markdown("""
        Random Forest Classifier
        Accuracy: 71.4%
        Trained on 215 keystrokes
        
        Features used:
        1. Typing speed
        2. Speed variation
        3. Minimum speed
        4. Maximum speed
        5. Long pauses
        """)
else:
    st.warning("Model file not found")

st.markdown("---")
st.subheader("Typing Test")

sample_text = """Machine learning is fascinating. I am testing an AI that detects attention drift from typing patterns. The system analyzes my typing speed and rhythm. When I lose focus, my typing pattern changes. This AI can help students stay focused while studying online."""

col1, col2 = st.columns([2, 1])

with col1:
    st.info("Copy this text:")
    st.code(sample_text, language="markdown")
    
    user_text = st.text_area(
        "Start typing here:",
        height=150,
        key="typing_area"
    )
    
    analyze_button = st.button("Analyze", type="primary")

def extract_features(text):
    if len(text) < 10:
        return None
    
    words = text.split()
    char_count = len(text)
    
    estimated_time = max(1, char_count / 5)
    avg_speed = char_count / estimated_time
    
    word_lengths = [len(w) for w in words] if words else [0]
    if len(word_lengths) > 1:
        std_speed = np.std(word_lengths)
    else:
        std_speed = 2.0
    
    if len(text) > 50:
        min_speed = min(avg_speed, 3.0)
        max_speed = max(avg_speed, 8.0)
    else:
        min_speed = 2.0
        max_speed = 6.0
    
    long_pauses = text.count('  ') + text.count('. ') + text.count('? ') + text.count('! ')
    
    return [avg_speed, std_speed, min_speed, max_speed, min(long_pauses, 10)]

if analyze_button:
    if len(user_text) < 30:
        st.warning("Please type at least 30 characters")
    else:
        with st.spinner("Analyzing..."):
            time.sleep(0.5)
            features = extract_features(user_text)
            
            with col2:
                st.markdown("### Result")
                
                if model_loaded and features:
                    features_array = np.array(features).reshape(1, -1)
                    prediction = model.predict(features_array)[0]
                    confidence = model.predict_proba(features_array)[0][prediction]
                    
                    if prediction == 1:
                        st.error("Distracted detected")
                        st.markdown(f"Confidence: {confidence:.1%}")
                        st.info("Suggestion: Take a short break")
                    else:
                        st.success("Focused")
                        st.markdown(f"Confidence: {confidence:.1%}")
                    
                    with st.expander("Details"):
                        st.markdown(f"""
                        Typing speed: {features[0]:.1f} chars/sec
                        Speed variation: {features[1]:.1f}
                        Long pauses: {features[4]}
                        Characters: {len(user_text)}
                        Words: {len(user_text.split())}
                        """)
                    
                    focus_score = (1 - confidence) if prediction == 1 else confidence
                    st.progress(focus_score)
                else:
                    st.warning("Model not available")

st.markdown("---")

if st.button("Save session"):
    if len(user_text) > 0:
        session_data = pd.DataFrame([{
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'text': user_text,
            'char_count': len(user_text),
            'word_count': len(user_text.split()),
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S")
        }])
        
        filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        session_data.to_csv(filename, index=False)
        st.success(f"Saved to {filename}")

st.markdown("---")
st.caption("Gunmay Parganiha | Graphic Era University")