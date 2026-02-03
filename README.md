# Hybrid AI Intrusion Detection System (IDS) 🛡️

**A real-time network security system leveraging Deep Learning (LSTM) and Machine Learning (Random Forest) to detect anomalies and cyberattacks.**

This project demonstrates the application of **Hybrid AI** architectures to cybersecurity. By combining the temporal sequence learning capabilities of **LSTMs** (for pattern recognition in traffic flows) with the classification speed of **Random Forests**, this system achieves high accuracy with low false-positive rates on PCAP data.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit_Learn-Random_Forest-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)

## 🧠 Key Features

* **Hybrid Detection Engine:** Uses a weighted ensemble of:
    * **LSTM (Long Short-Term Memory):** Captures sequential dependencies in TCP/IP packet flows (`src/lstm_model.h5`).
    * **Random Forest:** Analyzes statistical features of packet headers (`src/rf_model_fixed.pkl`).
* **Real-Time Monitoring:** Streamlit dashboard (`app.py`) visualizing traffic throughput and anomaly scores live.
* **Traffic Simulation:** Custom scripts (`scripts/simulate_network_traffic.py`) to generate benign and malicious traffic patterns for testing.
* **PCAP Analysis:** Raw packet processing pipeline using `scapy` and `pandas`.

## 📊 Performance Visualization

The system outputs confusion matrices and real-time alerts upon detecting signatures of known attacks (DoS, Probe, U2R).

*(You can add a screenshot of your Streamlit dashboard here if you have one)*

## 🛠 Tech Stack

* **Core:** Python, NumPy, Pandas
* **ML/DL:** TensorFlow (Keras), Scikit-learn
* **Networking:** Scapy (Packet crafting & parsing)
* **Visualization:** Streamlit, Matplotlib, Seaborn

## 📂 Project Structure

```bash
├── .streamlit/         # Dashboard config
├── scripts/            # Traffic simulation & data downloaders
├── src/
│   ├── models/         # Trained .h5 and .pkl models
│   ├── hybrid_model.py # Ensemble logic implementation
│   ├── detection_real_time.py # Live traffic analyzer
│   └── ...training     # Training scripts for RF and LSTM
├── app.py              # Main dashboard entry point
└── requirements.txt    # Dependencies
```

## 🚀 Running Locally

**Install dependencies**:

```bash
pip install -r requirements.txt
```

**Start the Dashboard**:

```bash
streamlit run app.py
```

**Simulate Traffic (Optional)**:

```bash
python scripts/simulate_network_traffic.py
```

---

Developed as part of MSc Computer Science research into AI-driven Cybersecurity.
