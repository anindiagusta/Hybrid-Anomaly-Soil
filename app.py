import streamlit as st
import numpy as np
import joblib



# PAGE CONFIG

st.set_page_config(
    page_title="Soil Anomaly Detection",
    page_icon="🌱",
    layout="wide"
)



# FAST MODEL LOADER (CACHED ONCE ONLY)

@st.cache_resource
def load_models():

    scaler = joblib.load("models/scaler_ae.pkl")
    knn = joblib.load("models/knn_k4.pkl")
    threshold = float(joblib.load("models/threshold.pkl"))

    scaler_ae = joblib.load("models/scaler_ae.pkl")
    scaler_knn = joblib.load("models/scaler_knn.pkl")

    return scaler, knn, threshold, scaler_ae, scaler_knn


scaler, knn_model, threshold, scaler_ae, scaler_knn = load_models()



# HEADER

st.title("🌱 Soil Anomaly Detection")
st.caption("Ultra Fast Hybrid kNN + Statistical AE Proxy")



# INPUT UI

col1, col2 = st.columns(2)

with col1:
    hu = st.number_input("Humidity", value=50.0)
    ta = st.number_input("Temperature", value=25.0)
    ec = st.number_input("EC", value=1.0)
    ph = st.number_input("pH", value=6.5)

with col2:
    n = st.number_input("Nitrogen", value=20.0)
    p = st.number_input("Phosphorus", value=15.0)
    k = st.number_input("Potassium", value=18.0)


run = st.button("🔍 Analyze", use_container_width=True)



# FAST INFERENCE (NO PANDAS)

if run:

    # ------------------------------------------
    # DIRECT NUMPY INPUT (FAST PATH)
    # ------------------------------------------
    x = np.array([[hu, ta, ec, ph, n, p, k]], dtype=np.float32)

    # ------------------------------------------
    # SCALING
    # ------------------------------------------
    x_scaled = scaler.transform(x)

    # ------------------------------------------
    # kNN SCORE (MAIN SIGNAL)
    # ------------------------------------------
    distances, _ = knn_model.kneighbors(x_scaled)
    knn_score = float(distances.mean())

    # ------------------------------------------
    # AE PROXY SCORE (DETERMINISTIC, NO NOISE)
    # ------------------------------------------
    ae_score = knn_score * 0.85

    # ------------------------------------------
    # NORMALIZATION
    # ------------------------------------------
    ae_norm = float(scaler_ae.transform([[ae_score]])[0][0])
    knn_norm = float(scaler_knn.transform([[knn_score]])[0][0])

    # ------------------------------------------
    # FUSION SCORE
    # ------------------------------------------
    fusion_score = 0.5 * ae_norm + 0.5 * knn_norm

    # ------------------------------------------
    # DECISION
    # ------------------------------------------
    is_anomaly = fusion_score > threshold

    status = "ANOMALY ⚠️" if is_anomaly else "NORMAL ✅"


    
    # OUTPUT (FAST UI)
    
    c1, c2, c3 = st.columns(3)

    c1.metric("Condition", status)
    c2.metric("Score", f"{fusion_score:.4f}")
    c3.metric("Threshold", f"{threshold:.4f}")


    st.markdown("---")

    
    # INSIGHT (LIGHTWEIGHT)
    
    st.subheader("💡 Insight")

    features = np.array([hu, ta, ec, ph, n, p, k])

    if np.all(features == 0):
        st.error("Sensor offline / no signal detected")

    elif np.any(features == 0):
        st.warning("Some sensor values missing or zero")

    elif not is_anomaly:
        st.success("System stable")

    else:
        idx = np.argmax(np.abs(features - features.mean()))
        labels = ["Humidity","Temperature","EC","pH","Nitrogen","Phosphorus","Potassium"]
        st.error(f"Main deviation: {labels[idx]}")


    
    # DEBUG (OPTIONAL)
    
    with st.expander("Debug"):
        st.write("kNN:", knn_score)
        st.write("AE proxy:", ae_score)
        st.write("AE norm:", ae_norm)
        st.write("kNN norm:", knn_norm)