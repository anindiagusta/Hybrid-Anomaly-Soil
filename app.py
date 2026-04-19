import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# TITLE
# =========================
st.title("🌱 Soil Sensor Anomaly Detection (Manual Input)")

# =========================
# LOAD DATA (BASELINE)
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_soil.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values('timestamp')
    df = df.fillna(method='ffill')
    return df

df = load_data()

# =========================
# PILIH SENSOR
# =========================
sensor_list = df['soil_sensor'].unique()
selected_sensor = st.selectbox("Pilih Sensor", sensor_list)

df_sensor = df[df['soil_sensor'] == selected_sensor].copy()

feature_cols = ['hu', 'ta', 'ec', 'ph', 'n', 'p', 'k']
X = df_sensor[feature_cols].values

# =========================
# NORMALISASI
# =========================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

input_dim = X_scaled.shape[1]

# =========================
# TRAIN AUTOENCODER (AE4)
# =========================
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
output = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, output)
autoencoder.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(patience=5, restore_best_weights=True)

with st.spinner("Training model..."):
    autoencoder.fit(
        X_scaled, X_scaled,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
        callbacks=[early_stop]
    )

# =========================
# KNN (K4)
# =========================
knn = NearestNeighbors(n_neighbors=5, metric='manhattan')
knn.fit(X_scaled)

# =========================
# HITUNG THRESHOLD DARI DATA TRAIN
# =========================
X_pred = autoencoder.predict(X_scaled, verbose=0)
recon_error = np.mean(np.square(X_scaled - X_pred), axis=1)

distances, _ = knn.kneighbors(X_scaled)
knn_distance = distances.mean(axis=1)

scaler_ae = MinMaxScaler()
scaler_knn = MinMaxScaler()

ae_score = scaler_ae.fit_transform(recon_error.reshape(-1,1))
knn_score = scaler_knn.fit_transform(knn_distance.reshape(-1,1))

fusion_score_train = 0.5 * ae_score + 0.5 * knn_score

threshold = np.quantile(fusion_score_train, 0.95)

st.success(f"Threshold (P95): {threshold:.4f}")

# =========================
# INPUT MANUAL
# =========================
st.subheader("✏️ Input Data Sensor")

col1, col2, col3 = st.columns(3)

hu = col1.number_input("Humidity (hu)", value=30.0)
ta = col2.number_input("Temperature (ta)", value=25.0)
ec = col3.number_input("EC", value=500)

ph = col1.number_input("pH", value=5.5)
n = col2.number_input("Nitrogen (n)", value=100)
p = col3.number_input("Phosphorus (p)", value=200)

k = st.number_input("Potassium (k)", value=150)

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Deteksi Anomali"):

    input_data = np.array([[hu, ta, ec, ph, n, p, k]])

    # scaling pakai scaler training
    input_scaled = scaler.transform(input_data)

    # AE error
    pred = autoencoder.predict(input_scaled, verbose=0)
    recon_err = np.mean(np.square(input_scaled - pred), axis=1)

    # KNN distance
    dist, _ = knn.kneighbors(input_scaled)
    knn_dist = dist.mean(axis=1)

    # normalisasi pakai scaler lama
    ae_norm = scaler_ae.transform(recon_err.reshape(-1,1))
    knn_norm = scaler_knn.transform(knn_dist.reshape(-1,1))

    fusion_score = 0.5 * ae_norm + 0.5 * knn_norm

    # =========================
    # HASIL
    # =========================
    st.subheader("📊 Hasil")

    st.write(f"Fusion Score: {fusion_score[0][0]:.4f}")

    if fusion_score > threshold:
        st.error("🚨 ANOMALI TERDETEKSI")
    else:
        st.success("✅ NORMAL")