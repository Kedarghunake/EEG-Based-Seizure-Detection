# app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="EEG Seizure Detection",
    page_icon="🧠",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

model = joblib.load("model_AE_5.pkl")

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}

.big-title {
    font-size: 45px;
    font-weight: bold;
    color: white;
    text-align: center;
}

.sub-title {
    font-size: 20px;
    color: #d1d1d1;
    text-align: center;
}

.metric-card {
    background-color: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

.result-normal {
    background-color: #00c853;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 25px;
    font-weight: bold;
}

.result-seizure {
    background-color: #d50000;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 25px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown('<p class="big-title">🧠 EEG Seizure Detection Dashboard</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="sub-title">Machine Learning Based Epileptic Seizure Detection using EEG Signals</p>',
    unsafe_allow_html=True
)

st.write("")

# ---------------- SIDEBAR ----------------

st.sidebar.title("📌 Project Info")

st.sidebar.info("""
### Models Used
- A vs E
- Random Forest Classifier

### Features Used
- Mean
- Standard Deviation
- Max
- Min
- RMS
- Energy

### Window Size
- 5 Seconds
""")

# ---------------- INPUT SECTION ----------------

st.subheader("📥 Enter EEG Feature Values")

col1, col2, col3 = st.columns(3)

with col1:
    mean = st.number_input("Mean", value=-10.0)

with col2:
    std = st.number_input("Standard Deviation", value=50.0)

with col3:
    mx = st.number_input("Maximum", value=120.0)

col4, col5, col6 = st.columns(3)

with col4:
    mn = st.number_input("Minimum", value=-130.0)

with col5:
    rms = st.number_input("RMS", value=55.0)

with col6:
    energy = st.number_input("Energy", value=150000.0)

# ---------------- FEATURE DISPLAY ----------------

features = np.array([[mean, std, mx, mn, rms, energy]])

st.write("")

# ---------------- PREDICTION BUTTON ----------------

if st.button("🔍 Predict EEG Condition"):

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0]

    st.write("")

    # ---------- RESULT ----------

    if prediction == 1:
        st.markdown(
            '<div class="result-seizure">⚠️ Seizure Detected</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-normal">✅ Normal EEG</div>',
            unsafe_allow_html=True
        )

    st.write("")

    # ---------- CONFIDENCE ----------

    col7, col8 = st.columns(2)

    with col7:
        st.metric("Normal Probability", f"{probability[0]*100:.2f}%")

    with col8:
        st.metric("Seizure Probability", f"{probability[1]*100:.2f}%")

    st.write("")

    # ---------- FEATURE VISUALIZATION ----------

    st.subheader("📊 Feature Visualization")

    feature_names = ["Mean", "Std", "Max", "Min", "RMS", "Energy"]
    feature_values = [mean, std, mx, mn, rms, energy]

    fig = go.Figure(
        data=[
            go.Bar(
                x=feature_names,
                y=feature_values
            )
        ]
    )

    fig.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- FOOTER ----------------

st.write("")
st.write("---")

st.markdown("""
<center>
Developed using Streamlit | EEG Seizure Detection Project
</center>
""", unsafe_allow_html=True)