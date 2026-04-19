import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

# =========================
# TITLE
# =========================
st.title("🌱 Soil Sensor Anomaly Detection (AE + kNN Fusion)")

# =========================
# UPLOAD DATA
# =========================
uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Preview Data:", df.head())

    # =========================
    # PREPROCESSING
    # =========================
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values('timestamp')
    df = df.fillna(method='ffill')

    sensors = df['soil_sensor'].unique()
    selected_sensor = st.selectbox("Pilih Sensor", sensors)

    df_sensor = df[df['soil_sensor'] == selected_sensor].copy()
    df_sensor = df_sensor.sort_values('timestamp').reset_index(drop=True)

    feature_cols = ['hu', 'ta', 'ec', 'ph', 'n', 'p', 'k']
    X = df_sensor[feature_cols].values

    # =========================
    # NORMALISASI
    # =========================
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    input_dim = X_scaled.shape[1]

    # =========================
    # AUTOENCODER (BEST MODEL AE4)
    # =========================
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    output = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(input_layer, output)
    autoencoder.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(patience=5, restore_best_weights=True)

    with st.spinner("Training Autoencoder..."):
        autoencoder.fit(
            X_scaled, X_scaled,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            callbacks=[early_stop]
        )

    # =========================
    # AE RECON ERROR
    # =========================
    X_pred = autoencoder.predict(X_scaled, verbose=0)
    recon_error = np.mean(np.square(X_scaled - X_pred), axis=1)

    # =========================
    # KNN (BEST MODEL K4)
    # =========================
    knn = NearestNeighbors(n_neighbors=5, metric='manhattan')
    knn.fit(X_scaled)

    distances, _ = knn.kneighbors(X_scaled)
    knn_distance = distances.mean(axis=1)

    # =========================
    # NORMALISASI SCORE
    # =========================
    scaler_ae = MinMaxScaler()
    scaler_knn = MinMaxScaler()

    ae_score = scaler_ae.fit_transform(recon_error.reshape(-1,1))
    knn_score = scaler_knn.fit_transform(knn_distance.reshape(-1,1))

    # =========================
    # FUSION SCORE
    # =========================
    fusion_score = 0.5 * ae_score + 0.5 * knn_score

    df_sensor['fusion_score'] = fusion_score

    # =========================
    # THRESHOLD (P95)
    # =========================
    threshold = np.quantile(fusion_score, 0.95)

    df_sensor['anomaly'] = np.where(
        df_sensor['fusion_score'] > threshold, 1, 0
    )

    # =========================
    # OUTPUT
    # =========================
    st.subheader("📊 Hasil Deteksi")

    st.write("Threshold:", threshold)

    anomaly_count = df_sensor['anomaly'].sum()
    st.write("Jumlah Anomali:", anomaly_count)

    st.dataframe(df_sensor)

    # =========================
    # PLOT
    # =========================
    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(df_sensor['timestamp'], df_sensor['fusion_score'], label='Fusion Score')
    ax.axhline(threshold, color='orange', linestyle='--', label='Threshold')

    anomalies = df_sensor[df_sensor['anomaly'] == 1]

    ax.scatter(
        anomalies['timestamp'],
        anomalies['fusion_score'],
        color='red',
        label='Anomaly'
    )

    ax.legend()
    ax.set_title("Anomaly Detection")
    ax.grid()

    st.pyplot(fig)