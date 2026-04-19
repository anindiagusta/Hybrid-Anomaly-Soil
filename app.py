# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==================================================
# CONFIG PAGE
# ==================================================
st.set_page_config(
    page_title="Soil Anomaly Detection",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Soil Sensor Anomaly Detection")
st.caption("Hybrid Model: Autoencoder + kNN")

# ==================================================
# LOAD FILE MODEL
# ==================================================
scaler = joblib.load("models/scaler.pkl")
knn_model = joblib.load("models/knn_model.pkl")
threshold = joblib.load("models/threshold.pkl")

# ==================================================
# INPUT USER
# ==================================================
st.subheader("📥 Input Data Sensor")

col1, col2 = st.columns(2)

with col1:
    hu = st.number_input("Humidity (hu)", value=50.0)
    ta = st.number_input("Temperature (ta)", value=25.0)
    ec = st.number_input("EC", value=1.0)
    ph = st.number_input("pH", value=6.5)

with col2:
    n = st.number_input("Nitrogen (n)", value=20.0)
    p = st.number_input("Phosphorus (p)", value=15.0)
    k = st.number_input("Potassium (k)", value=18.0)

# ==================================================
# PREDICT BUTTON
# ==================================================
if st.button("🔍 Detect"):

    # ----------------------------------------------
    # DATAFRAME INPUT
    # ----------------------------------------------
    input_data = pd.DataFrame([{
        "hu": hu,
        "ta": ta,
        "ec": ec,
        "ph": ph,
        "n": n,
        "p": p,
        "k": k
    }])

    # ----------------------------------------------
    # SCALING
    # ----------------------------------------------
    X_scaled = scaler.transform(input_data)

    # ----------------------------------------------
    # kNN SCORE
    # ----------------------------------------------
    distances, _ = knn_model.kneighbors(X_scaled)
    knn_score = distances.mean(axis=1)[0]

    # ----------------------------------------------
    # SIMPLE SCORE (sementara)
    # ----------------------------------------------
    fusion_score = knn_score

    # ----------------------------------------------
    # STATUS
    # ----------------------------------------------
    if fusion_score > threshold:
        status = "ANOMALY ⚠️"
        flag = 1
    else:
        status = "NORMAL ✅"
        flag = 0

    # ==================================================
    # OUTPUT
    # ==================================================
    st.subheader("📊 Prediction Result")

    st.metric("Status", status)
    st.metric("Score", round(fusion_score, 4))
    st.metric("Threshold", round(threshold, 4))

    # ==================================================
    # ROOT CAUSE ANALYSIS
    # ==================================================
    st.subheader("🧠 Root Cause Analysis")

    if flag == 0:
        st.success("Sensor condition is normal.")
    else:

        values = {
            "Humidity": hu,
            "Temperature": ta,
            "EC": ec,
            "pH": ph,
            "Nitrogen": n,
            "Phosphorus": p,
            "Potassium": k
        }

        # cari nilai paling ekstrem sederhana
        arr = np.array(list(values.values()))
        idx = np.argmax(np.abs(arr - np.mean(arr)))

        cause = list(values.keys())[idx]

        st.error(f"Main Cause: {cause} abnormal value detected.")

    # ==================================================
    # SHOW INPUT DATA
    # ==================================================
    st.subheader("📋 Input Summary")
    st.dataframe(input_data, use_container_width=True)