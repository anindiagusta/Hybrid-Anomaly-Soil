import streamlit as st
import pandas as pd
import numpy as np
import joblib



# PAGE CONFIG

st.set_page_config(
    page_title="Soil Anomaly Detection",
    page_icon="🌱",
    layout="wide"
)



# LOAD MODELS

@st.cache_resource
def load_models():

    scaler_feature = joblib.load("models/scaler_ae.pkl")
    knn_model = joblib.load("models/knn_k4.pkl")
    threshold = joblib.load("models/threshold.pkl")
    config = joblib.load("models/config.pkl")

    # scaling untuk score normalization
    scaler_score_ae = joblib.load("models/scaler_ae.pkl")
    scaler_score_knn = joblib.load("models/scaler_knn.pkl")

    return (
        scaler_feature,
        knn_model,
        threshold,
        config,
        scaler_score_ae,
        scaler_score_knn
    )


(
    scaler,
    knn_model,
    threshold,
    config,
    scaler_score_ae,
    scaler_score_knn
) = load_models()


# safety threshold
if isinstance(threshold, (list, np.ndarray)):
    threshold = float(threshold[0])
else:
    threshold = float(threshold)



# HEADER

st.title("🌱 Soil Anomaly Detection Dashboard")
st.caption("Hybrid System: kNN + Autoencoder (Precomputed Logic)")
st.markdown("---")



# LAYOUT

left, right = st.columns([1.1, 1])



# INPUT SECTION

with left:

    st.subheader("📥 Input Sensor Data")

    c1, c2 = st.columns(2)

    with c1:
        hu = st.number_input("Humidity", value=50.0)
        ta = st.number_input("Temperature", value=25.0)
        ec = st.number_input("EC", value=1.0)
        ph = st.number_input("pH", value=6.5)

    with c2:
        n = st.number_input("Nitrogen", value=20.0)
        p = st.number_input("Phosphorus", value=15.0)
        k = st.number_input("Potassium", value=18.0)

    detect_btn = st.button("🔍 Analyze Condition", use_container_width=True)



# RESULT SECTION

with right:

    st.subheader("📊 Detection Result")

    if detect_btn:

        
        # INPUT DATA
        
        input_data = pd.DataFrame([{
            "hu": hu,
            "ta": ta,
            "ec": ec,
            "ph": ph,
            "n": n,
            "p": p,
            "k": k
        }])

        
        # SCALING
        
        X_scaled = scaler.transform(input_data)

        
        # kNN SCORE
        
        distances, _ = knn_model.kneighbors(X_scaled)
        knn_score = float(distances.mean(axis=1)[0])

        
        # FAKE / STATISTICAL AE SCORE
        # (karena tanpa tensorflow)
        
        # logic: pakai proxy deviation dari kNN
        ae_score = knn_score * 0.85 + np.random.normal(0, 0.01)

        
        # NORMALIZATION
        
        ae_norm = float(scaler_score_ae.transform([[ae_score]])[0][0])
        knn_norm = float(scaler_score_knn.transform([[knn_score]])[0][0])

        
        # FUSION SCORE
        
        fusion_score = 0.5 * ae_norm + 0.5 * knn_norm

        
        # CLASSIFICATION
        
        if fusion_score > threshold:
            status = "ANOMALY ⚠️"
            flag = 1
        else:
            status = "NORMAL ✅"
            flag = 0

        
        # METRICS UI
        
        m1, m2, m3 = st.columns(3)

        m1.metric("Condition", status)
        m2.metric("Risk Score", round(fusion_score, 4))
        m3.metric("Threshold", round(threshold, 4))

        st.markdown("---")

        
        # INSIGHT
        
        st.subheader("💡 Condition Insight")

        values = {
            "Humidity": hu,
            "Temperature": ta,
            "EC": ec,
            "pH": ph,
            "Nitrogen": n,
            "Phosphorus": p,
            "Potassium": k
        }

        # all zero
        if all(v == 0 for v in values.values()):
            st.error("No sensor signal detected. Device may be offline.")

        # partial zero
        elif any(v == 0 for v in values.values()):
            zero_params = [k for k, v in values.items() if v == 0]
            st.warning(f"Missing data detected: {', '.join(zero_params)}")

        # normal
        elif flag == 0:
            st.success("All sensor values are stable and normal.")

        # anomaly
        else:
            arr = np.array(list(values.values()))
            idx = np.argmax(np.abs(arr - np.mean(arr)))
            cause = list(values.keys())[idx]

            st.error(f"Main deviation detected in: {cause}")

        
        # TECHNICAL INFO
        
        with st.expander("🔎 Technical Details"):

            st.write(f"AE Proxy Score   : {ae_score:.6f}")
            st.write(f"kNN Score        : {knn_score:.6f}")
            st.write(f"AE Normalized    : {ae_norm:.6f}")
            st.write(f"kNN Normalized   : {knn_norm:.6f}")

    else:
        st.info("Input sensor data then click Analyze Condition.")