import streamlit as st
import numpy as np
import joblib
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="🌱",
    layout="wide"
)

# =====================================================
# STYLE
# =====================================================
st.markdown("""
<style>
.block-container{
    padding:1rem 2rem;
}

/* input kecil */
input[type=number]{
    height:35px;
    font-size:13px;
}

/* title */
.main-title{
    font-size:34px;
    font-weight:800;
    background:linear-gradient(90deg,#16a34a,#0284c7);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.sub-title{
    color:#64748b;
    margin-bottom:10px;
}

/* metric */
.metric-box{
    background:#f8fafc;
    border-radius:14px;
    padding:12px;
    text-align:center;
}

.small{
    font-size:12px;
    color:#64748b;
}

/* button */
.stButton > button{
    width:100%;
    background:linear-gradient(90deg,#16a34a,#0284c7);
    color:white;
    border:none;
    border-radius:10px;
    font-weight:700;
    padding:0.6rem;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.write("")
st.markdown("<div class='main-title'>🌱 IoT Sensor Anomaly Detection Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>K-Nearest Neighbors Based Anomaly Classification for Soil Sensor Data</div>", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_models():
    model_path = "models/"
    scaler = joblib.load(os.path.join(model_path, "knn_scaler.pkl"))
    knn_model = joblib.load(os.path.join(model_path, "knn_model.pkl"))
    return scaler, knn_model

try:
    scaler, knn_model = load_models()
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    missing_file = str(e)

# =====================================================
# DEFAULT DATA
# =====================================================
manual_tests = {
    "S1": [33.5, 25.6, 650, 5.0, 108, 295, 288],
    "S2": [41.2, 24.9, 410, 5.4, 72, 210, 202],
    "S3": [8.5, 12.0, 80, 3.2, 8, 15, 12],
    "S4": [33.5, 25.6, 40, 5.0, 108, 295, 288],
    "S5": [80, 38, 1200, 8.5, 300, 500, 450],
    "S6": [12, 18, 90, 3.5, 5, 20, 15],
}

sensor_data = [hu, ta, ec, ph, n, p, k]

if any(v <= 0 for v in sensor_data):
    st.warning(
        "⚠️ Semua parameter harus lebih besar dari 0."
    )
    st.stop()

features = ["hu", "ta", "ec", "ph", "n", "p", "k"]
sensor_ids = list(manual_tests.keys())

# =====================================================
# LAYOUT
# =====================================================
left, right = st.columns([1.4, 1])

# =====================================================
# INPUT TABLE
# =====================================================
with left:

    st.markdown("### Sensor Input")

    header = st.columns(8)
    header[0].markdown("**Sensor**")
    for i, f in enumerate(features):
        header[i+1].markdown(f"**{f.upper()}**")

    sensor_data = []

    for s in sensor_ids:

        cols = st.columns(8)
        cols[0].markdown(f"**{s}**")

        row = []
        for i, f in enumerate(features):
            val = cols[i+1].number_input(
                "",
                value=float(manual_tests[s][i]),
                key=f"{s}_{f}",
                label_visibility="collapsed"
            )
            row.append(val)

        sensor_data.append(row)

    run = st.button("Analyze Now", use_container_width=True)

# =====================================================
# DEFAULT RIGHT
# =====================================================
with right:
    if not run:
        st.markdown("### Analysis Result")
        if not model_loaded:
            st.error(f"❌ Model file not found: {missing_file}")
            st.info("Pastikan `knn_scaler.pkl` dan `knn_model.pkl` ada di direktori yang sama.")
        else:
            st.info("Waiting for input...")

# =====================================================
# CLASSIFICATION
# =====================================================
if run:

    if not model_loaded:
        with right:
            st.markdown("### Analysis Result")
            st.error(f"❌ Model file not found: {missing_file}")
            st.info("Pastikan `knn_scaler.pkl` dan `knn_model.pkl` ada di direktori yang sama.")
    else:
        X = np.array(sensor_data)

        # Scale & Predict
        X_scaled = scaler.transform(X)
        predictions = knn_model.predict(X_scaled)

        # predictions: 1 = normal, -1 atau 0 = anomaly
        # Sesuaikan logika label tergantung encoding model kamu
        # Jika model menggunakan label: "normal" / "anomaly" (string), sesuaikan di bawah
        anomaly_idx = set()
        pred_labels = []

        for i, pred in enumerate(predictions):
            # Mendukung label numerik (-1/1) maupun string ("anomaly"/"normal")
            if pred == -1 or str(pred).lower() in ("anomaly", "0", "false"):
                anomaly_idx.add(i)
                pred_labels.append("anomaly")
            else:
                pred_labels.append("normal")

        flag = len(anomaly_idx) > 0

        # =====================================================
        # OUTPUT
        # =====================================================
        with right:

            st.markdown("### Analysis Result")

            if flag:
                st.error("⚠️ Anomaly Detected")
            else:
                st.success("✅ All Sensors Normal")

            # SENSOR STATUS
            st.markdown("### Sensor Status")
            cols = st.columns(2)

            for i in range(len(sensor_ids)):
                if i in anomaly_idx:
                    cols[i % 2].warning(f"{sensor_ids[i]} anomaly")
                else:
                    cols[i % 2].success(f"{sensor_ids[i]} normal")

            # # DETAIL
            # st.markdown("### Detailed Insight")

            # if flag:
            #     for i in anomaly_idx:
            #         st.write("•", f"{sensor_ids[i]} terdeteksi sebagai anomaly oleh model KNN")
            # else:
            #     st.write("All sensors consistent.")