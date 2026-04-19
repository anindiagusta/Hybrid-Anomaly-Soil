# 🌱 Hybrid Soil Anomaly Detection System

A **Hybrid Machine Learning system** for detecting anomalies in soil sensor data using a combination of **Autoencoder (offline training)** and **k-Nearest Neighbors (kNN)** with a **Streamlit-based interactive dashboard**.

---

## 🚀 Features

- 📊 Anomaly detection
- 🤖 Hybrid ML approach (Autoencoder + kNN)
- ⚡ Lightweight deployment (no heavy deep learning runtime)
- 🧠 Intelligent condition scoring system
- 💡 Simple anomaly insight / explanation module
- 🌐 Interactive Streamlit dashboard

---

## 📥 Input Parameters

The system analyzes soil conditions based on:

- Humidity (hu)
- Temperature (ta)
- Electrical Conductivity (EC)
- pH level
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)

---

## ⚙️ System Workflow

1. User inputs sensor data
2. Data is normalized using pre-trained scaler
3. kNN computes distance-based anomaly score
4. Autoencoder score (proxy / pretrained logic) is used
5. Fusion score is calculated
6. Thresholding determines:
   - NORMAL condition
   - ANOMALY condition
7. System provides simple explanation of anomaly source

---

## 📊 Output

The dashboard displays:

- Condition Status (Normal / Anomaly)
- Risk Score (Fusion Score)
- Threshold Limit
- Technical breakdown (AE + kNN score)
- Simple anomaly insight (affected parameter)

---

## 🧠 Model Architecture

- Autoencoder (trained offline)
- k-Nearest Neighbors (kNN)
- MinMaxScaler normalization
- Fusion-based scoring system

> Note: Deployment version does not require TensorFlow.

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas & NumPy
- Joblib

---

## ⚡ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py