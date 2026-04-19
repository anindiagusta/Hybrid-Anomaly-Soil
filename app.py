import streamlit as st
import numpy as np
import joblib
import os

# =========================
# TITLE
# =========================
st.title("🌱 Soil Sensor Anomaly Detection (kNN Only)")

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "models/"

required_files = [
    "knn_k4.pkl",
    "scaler_knn.pkl",
    "threshold_knn.pkl",
    "config.pkl"
]

missing = [f for f in required_files if not os.path.exists(MODEL_PATH + f)]

if missing:
    st.error(f"❌ File tidak ditemukan: {missing}")
    st.stop()

knn = joblib.load(MODEL_PATH + "knn_k4.pkl")
scaler = joblib.load(MODEL_PATH + "scaler_knn.pkl")
threshold = joblib.load(MODEL_PATH + "threshold_knn.pkl")
config = joblib.load(MODEL_PATH + "config.pkl")

st.success("✅ Model kNN berhasil dimuat")

# =========================
# INPUT MANUAL
# =========================
st.subheader("✏️ Input Data Sensor")

col1, col2, col3 = st.columns(3)

hu = col1.number_input("Humidity (hu)", value=30.0)
ta = col2.number_input("Temperature (ta)", value=25.0)
ec = col3.number_input("EC", value=500.0)

ph = col1.number_input("pH", value=5.5)
n = col2.number_input("Nitrogen (n)", value=100.0)
p = col3.number_input("Phosphorus (p)", value=200.0)

k = st.number_input("Potassium (k)", value=150.0)

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Deteksi Anomali"):

    input_data = np.array([[hu, ta, ec, ph, n, p, k]])

    # scaling
    input_scaled = scaler.transform(input_data)

    # hitung distance ke kNN
    distances, _ = knn.kneighbors(input_scaled)
    knn_score = distances.mean(axis=1)

    # =========================
    # OUTPUT
    # =========================
    st.subheader("📊 Hasil")

    st.write(f"kNN Distance Score: {knn_score[0]:.6f}")
    st.write(f"Threshold (P95): {threshold:.6f}")

    if knn_score > threshold:
        st.error("🚨 ANOMALI TERDETEKSI")
    else:
        st.success("✅ NORMAL")