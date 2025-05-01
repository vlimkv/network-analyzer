
# Intelligent Network Traffic Analyzer

A professional web-based solution for detecting anomalies and potential intrusions in network traffic using hybrid machine learning models: Random Forest and LSTM. The interface is built with Streamlit and supports large-scale data, real-time inspection, and comprehensive evaluation metrics.

---

## 🔍 Project Overview

This system is designed to analyze structured network traffic data and detect intrusions based on behavior patterns. The solution combines:

- **Random Forest (RF)** – effective with structured tabular features.
- **Long Short-Term Memory (LSTM)** – captures temporal behavior in traffic.
- **Hybrid Model** – averages predictions from both models for increased robustness.

It provides an intuitive web UI for loading traffic files, triggering analysis, and reviewing results.

---

## ✅ Key Features

- **Interactive Web UI** (Streamlit-powered)
- **CSV Upload** – supports files >1GB
- **Hybrid ML Prediction**: RF + LSTM
- **Result Table** with anomaly flags
- **Classification Metrics**: accuracy, precision, recall, F1-score
- **Visualizations**: confusion matrices, ROC curves, AUC
- **Batch & Real-Time Mode**

---

## 📁 Project Structure

```
project_root/
├── app.py                       # Streamlit web interface
├── requirements.txt             # Python dependencies
├── models/                      # Pretrained ML models
│   ├── rf_model_fixed.pkl       # RandomForestClassifier (joblib)
│   ├── lstm_model.h5            # LSTM (Keras model)
│   └── scaler_lstm.npy          # Normalization mean (NumPy)
├── data/                        # Example CSV input
│   └── example.csv
├── unzipped_system/             # Metrics and evaluation artifacts
│   ├── confusion_matrix_*.png
│   ├── roc_curve_*.png
│   └── classification_report_*.csv
└── .streamlit/config.toml       # Streamlit config (increased upload/message size)
```

---

## ⚙️ Setup & Run

### 1. Install environment
```bash
python -m venv venv
source venv/bin/activate     # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Configure Streamlit
Ensure the following config exists at `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 1024
maxMessageSize = 1024
```

### 3. Start the application
```bash
streamlit run app.py
```

---

## 📤 Input Data Format

CSV file with at least the following columns:

```csv
duration,src_bytes,dst_bytes,wrong_fragment,label
0.01,100,200,0,0
0.30,1500,4000,0,1
...
```

- `label` must be `0` (normal) or `1` (attack)
- This column is required for calculating metrics

---

## 📊 Metrics & Evaluation

- Classification reports are generated live (via `sklearn.metrics`)
- Visualizations:
  - Confusion Matrices (RF, LSTM, Hybrid)
  - ROC Curves with AUC
- All assets stored in `/unzipped_system/`

---

## 🧠 Model Notes

- **RF model** was trained on tabular numerical features
- **LSTM model** was trained on normalized traffic data with time-dependencies
- **scaler_lstm.npy** stores the original mean used for normalization

---

## 📚 Technologies Used

- Python, NumPy, Pandas
- Scikit-learn, TensorFlow/Keras
- Streamlit (web UI)
- Matplotlib, Seaborn (visualization)

---

## 📌 License & Usage

This code is intended for academic, research, or demonstration purposes. For production use, ensure:
- Continuous retraining on up-to-date traffic
- Use of secure data ingestion pipelines
- Load balancing and model optimization at scale

---

## 👨‍💻 Author

Developed by **Alimkhan Slambek**  
Astana IT University  
Master’s Thesis: *Intelligent approaches for determining penetration into an organization's corporate network*
