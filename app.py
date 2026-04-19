import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Soil Anomaly Detection",
    page_icon="🌱",
    layout="wide"
)

# ==========================================================
# LOAD MODELS
# ==========================================================
@st.cache_resource
def load_models():

    scaler = joblib.load("models/scaler_ae.pkl")
    knn_model = joblib.load("models/knn_k4.pkl")
    threshold = joblib.load("models/threshold.pkl")
    config = joblib.load("models/config.pkl")

    # scaler untuk normalisasi score
    scaler_score_ae = joblib.load("models/scaler_score_ae.pkl")
    scaler_score_knn = joblib.load("models/scaler_knn.pkl")

    # AE MODEL (KERAS)
    ae_model = load_model("models/ae4_model.keras", compile=False)

    return scaler, knn_model, threshold, config, ae_model, scaler_score_ae, scaler_score_knn


(
    scaler,
    knn_model,
    threshold,
    config,
    ae_model,
    scaler_score_ae,
    scaler_score_knn
) = load_models()

# threshold safety
threshold = float(threshold[0]) if isinstance(threshold, (list, np.ndarray)) else float(threshold)

# ==========================================================
# HEADER
# ==========================================================
st.title("🌱 Soil Anomaly Detection Dashboard")
st.caption("Hybrid Keras Autoencoder + kNN System")
st.markdown("---")

# ==========================================================
# LAYOUT
# ==========================================================
left, right = st.columns([1.2, 1, 1])

# ==========================================================
# INPUT
# ==========================================================
with left:
    st.subheader("📥 Sensor Input")

    hu = st.number_input("Humidity", value=50.0)
    ta = st.number_input("Temperature", value=25.0)
    ec = st.number_input("EC", value=1.0)
    ph = st.number_input("pH", value=6.5)
    n = st.number_input("Nitrogen", value=20.0)
    p = st.number_input("Phosphorus", value=15.0)
    k = st.number_input("Potassium", value=18.0)

    run = st.button("⚡ Analyze", use_container_width=True)

# ==========================================================
# PROCESS
# ==========================================================
if run:

    input_data = np.array([[hu, ta, ec, ph, n, p, k]])

    # --------------------------
    # SCALING INPUT
    # --------------------------
    X_scaled = scaler.transform(input_data)

    # --------------------------
    # AE SCORE (REAL KERAS AE)
    # --------------------------
    X_pred = ae_model.predict(X_scaled, verbose=0)

    ae_score = np.mean(np.square(X_scaled - X_pred), axis=1)[0]

    # --------------------------
    # kNN SCORE
    # --------------------------
    distances, _ = knn_model.kneighbors(X_scaled)
    knn_score = distances.mean()

    # --------------------------
    # NORMALIZATION SCORE
    # --------------------------
    ae_norm = scaler_score_ae.transform([[ae_score]])[0][0]
    knn_norm = scaler_score_knn.transform([[knn_score]])[0][0]

    # --------------------------
    # FUSION SCORE
    # --------------------------
    fusion_score = 0.5 * ae_norm + 0.5 * knn_norm

    # --------------------------
    # CLASSIFICATION
    # --------------------------
    is_anomaly = fusion_score > threshold
    status = "⚠️ ANOMALY" if is_anomaly else "✅ NORMAL"

    # ==========================================================
    # METRICS
    # ==========================================================
    with right:
        st.subheader("📊 Result")

        st.metric("Condition", status)
        st.metric("Fusion Score", f"{fusion_score:.4f}")
        st.metric("Threshold", f"{threshold:.4f}")

    # ==========================================================
    # INSIGHT
    # ==========================================================
    st.markdown("---")
    st.subheader("💡 Insight")

    values = {
        "Humidity": hu,
        "Temperature": ta,
        "EC": ec,
        "pH": ph,
        "Nitrogen": n,
        "Phosphorus": p,
        "Potassium": k
    }

    if all(v == 0 for v in values.values()):
        st.error("Device offline / power failure detected")

    elif any(v == 0 for v in values.values()):
        missing = [k for k, v in values.items() if v == 0]
        st.warning(f"Sensor failure detected: {', '.join(missing)}")

    elif not is_anomaly:
        st.success("All sensor readings are stable")

    else:
        arr = np.array(list(values.values()))
        idx = np.argmax(np.abs(arr - np.mean(arr)))
        cause = list(values.keys())[idx]
        st.error(f"Main anomaly source: {cause}")

    # ==========================================================
    # TECHNICAL DETAILS
    # ==========================================================
    with st.expander("🔎 Technical Details"):
        st.write(f"AE Score: {ae_score:.6f}")
        st.write(f"kNN Score: {knn_score:.6f}")
        st.write(f"AE Normalized: {ae_norm:.6f}")
        st.write(f"kNN Normalized: {knn_norm:.6f}")

else:
    st.info("Input sensor data then click Analyze")