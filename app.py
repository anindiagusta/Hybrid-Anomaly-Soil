import streamlit as st
import numpy as np
import joblib
import os

from tensorflow.keras.models import load_model

# =========================
# TITLE
# =========================
st.title("🌱 Soil Sensor Anomaly Detection (Production Ready)")

# =========================
# LOAD MODEL (SAFE)
# =========================
MODEL_PATH = "models/"

required_files = [
    "ae4_model.h5",
    "scaler_ae.pkl",
    "knn_k4.pkl",
    "scaler_knn.pkl",
    "threshold.pkl",
    "config.pkl"
]

missing_files = [f for f in required_files if not os.path.exists(MODEL_PATH + f)]

if missing_files:
    st.error(f"❌ File tidak ditemukan: {missing_files}")
    st.stop()

# =========================
# LOAD ALL
# =========================
autoencoder = load_model(MODEL_PATH + "ae4_model.h5")
scaler_ae = joblib.load(MODEL_PATH + "scaler_ae.pkl")
knn = joblib.load(MODEL_PATH + "knn_k4.pkl")
scaler_knn = joblib.load(MODEL_PATH + "scaler_knn.pkl")
threshold = joblib.load(MODEL_PATH + "threshold.pkl")
config = joblib.load(MODEL_PATH + "config.pkl")

feature_cols = config["features"]

st.success("✅ Model berhasil dimuat")

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
# PREDICT
# =========================
if st.button("🔍 Deteksi Anomali"):

    input_data = np.array([[hu, ta, ec, ph, n, p, k]])

    # =========================
    # SCALING (PAKAI SCALER TRAINING)
    # =========================
    input_scaled = scaler_ae.transform(input_data)

    # =========================
    # AUTOENCODER
    # =========================
    pred = autoencoder.predict(input_scaled, verbose=0)
    recon_error = np.mean(np.square(input_scaled - pred), axis=1)

    # =========================
    # KNN
    # =========================
    input_scaled_knn = scaler_knn.transform(input_data)
    distances, _ = knn.kneighbors(input_scaled_knn)
    knn_distance = distances.mean(axis=1)

    # =========================
    # NORMALISASI SCORE
    # =========================
    ae_score = recon_error.reshape(-1,1)
    knn_score = knn_distance.reshape(-1,1)

    # IMPORTANT: pakai scaler yang sama
    ae_norm = ae_score  # sudah sesuai training
    knn_norm = knn_score

    # =========================
    # FUSION
    # =========================
    fusion_score = (
        config["fusion_weight_ae"] * ae_norm +
        config["fusion_weight_knn"] * knn_norm
    )

    # =========================
    # OUTPUT
    # =========================
    st.subheader("📊 Hasil")

    st.write(f"Fusion Score: {fusion_score[0][0]:.6f}")
    st.write(f"Threshold: {threshold:.6f}")

    if fusion_score > threshold:
        st.error("🚨 ANOMALI TERDETEKSI")
    else:
        st.success("✅ NORMAL")