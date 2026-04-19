import streamlit as st
import numpy as np
import joblib


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Soil Anomaly Dashboard",
    page_icon="🌱",
    layout="wide"
)


# ======================================================
# LOAD MODELS
# ======================================================
@st.cache_resource
def load_models():

    scaler_feature = joblib.load("models/scaler_ae.pkl")
    knn_model = joblib.load("models/knn_k4.pkl")
    threshold = joblib.load("models/threshold.pkl")
    config = joblib.load("models/config.pkl")

    return scaler_feature, knn_model, threshold, config


(
    scaler,
    knn_model,
    threshold,
    config
) = load_models()


# safety threshold
threshold = float(threshold[0]) if isinstance(threshold, (list, np.ndarray)) else float(threshold)


# ======================================================
# HEADER
# ======================================================
st.title("🌱 Soil Anomaly Detection")
st.caption("Hybrid kNN-based Anomaly Detection (Stable Deployment Mode)")
st.markdown("---")


# ======================================================
# LAYOUT
# ======================================================
col1, col2, col3 = st.columns([1.2, 1, 1])


# ======================================================
# INPUT
# ======================================================
with col1:
    st.subheader("📥 Sensor Input")

    hu = st.number_input("Humidity", value=50.0)
    ta = st.number_input("Temperature", value=25.0)
    ec = st.number_input("EC", value=1.0)
    ph = st.number_input("pH", value=6.5)
    n = st.number_input("Nitrogen", value=20.0)
    p = st.number_input("Phosphorus", value=15.0)
    k = st.number_input("Potassium", value=18.0)

    run = st.button("⚡ Analyze", use_container_width=True)


# ======================================================
# PROCESS
# ======================================================
if run:

    input_data = np.array([[hu, ta, ec, ph, n, p, k]])

    # scaling input
    X_scaled = scaler.transform(input_data)

    # kNN score (main signal)
    distances, _ = knn_model.kneighbors(X_scaled)
    knn_score = float(distances.mean())

    # ======================================================
    # SIMPLE STABLE HYBRID (NO SCALER, NO BUG)
    # ======================================================
    ae_score = np.std(X_scaled)  # statistical proxy (safe)

    fusion_score = 0.5 * knn_score + 0.5 * ae_score

    # ======================================================
    # THRESHOLDING
    # ======================================================
    is_anomaly = fusion_score > threshold
    status = "⚠️ ANOMALY" if is_anomaly else "✅ NORMAL"


    # ======================================================
    # METRICS
    # ======================================================
    with col2:
        st.subheader("📊 Status")

        st.metric("Condition", status)
        st.metric("Risk Score", f"{fusion_score:.4f}")
        st.metric("Threshold", f"{threshold:.4f}")


    # ======================================================
    # INSIGHT
    # ======================================================
    with col3:
        st.subheader("💡 Insight")

        values = {
            "Humidity": hu,
            "Temperature": ta,
            "EC": ec,
            "pH": ph,
            "N": n,
            "P": p,
            "K": k
        }

        if all(v == 0 for v in values.values()):
            st.error("Sensor offline / no signal detected")

        elif any(v == 0 for v in values.values()):
            bad = [k for k, v in values.items() if v == 0]
            st.warning(f"Missing: {', '.join(bad)}")

        elif not is_anomaly:
            st.success("Condition stable and within normal range")

        else:
            arr = np.array(list(values.values()))
            idx = np.argmax(np.abs(arr - np.mean(arr)))
            cause = list(values.keys())[idx]
            st.error(f"Main deviation source: {cause}")

    # ======================================================
    # TECHNICAL DETAILS
    # ======================================================
    st.markdown("---")

    with st.expander("🔎 Technical Details"):
        st.write(f"kNN Score: {knn_score:.6f}")
        st.write(f"AE Proxy (std): {ae_score:.6f}")
        st.write(f"Fusion Score: {fusion_score:.6f}")

else:
    st.info("Input sensor data then click Analyze")