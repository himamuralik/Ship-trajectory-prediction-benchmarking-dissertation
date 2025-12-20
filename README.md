# 🚢 Ship Trajectory Prediction using Deep Learning (AIS Data)

![Python](https://img.shields.io/badge/Python-3.9-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Status](https://img.shields.io/badge/Status-Completed-green)

### 📖 Executive Summary
This project conducts a comprehensive comparative analysis of Recurrent Neural Networks (RNNs) for maritime navigation safety. Using **2GB+ of AIS data** from New York Harbor, I engineered an end-to-end pipeline to predict future vessel coordinates with high geospatial precision.

**Key Technical Achievements:**
* **Advanced Modeling:** Benchmarked **LSTM, GRU, BiLSTM, and BiLSTM-Attention** architectures.
* **Key Finding:** **BiLSTM-Attention** achieved the highest accuracy for complex maneuvers, while **GRU** offered the best latency (8ms) for real-time systems.
* **ETL Optimization:** Automated the data preprocessing pipeline, reducing execution time by **85% (from 6 hours to 45 mins)** for 5M+ records using sliding window sequencing.
* **Custom Metrics:** Implemented **Haversine Distance Loss** to minimize physical error in kilometers rather than just statistical MSE.

---

### 🛠️ Tech Stack
* **Language:** Python 3.9
* **Deep Learning:** TensorFlow, Keras (Custom Layers for Attention Mechanisms)
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Geospatial & Viz:** Folium, Matplotlib, Seaborn

---

### 📊 Performance Benchmarks
All models were trained on the same New York Harbor dataset with a look-back window of 10 timestamps.

| Model Architecture | MSE Loss | Haversine Error (km) | Inference Time (ms) |
| :--- | :--- | :--- | :--- |
| LSTM | 0.0042 | 0.15 | 12 |
| GRU | 0.0038 | 0.14 | **8** (Best Latency) |
| BiLSTM | 0.0035 | 0.12 | 18 |
| **BiLSTM-Attention** | **0.0029** | **0.09** (Best Accuracy) | 22 |

---

### 🧩 System Architecture

#### 1. Data Ingestion & Preprocessing
The raw AIS data (MMSI, Latitude, Longitude, SOG, COG) contained noise and irregular timestamps.
* **Cleaning:** Filtered out stationary vessels (SOG < 0.5 knots) and anomalous GPS jumps.
* **Normalization:** MinMax scaling applied to coordinates to ensure stable gradient descent.
* **Sequence Generation:** Created a sliding window dataset ($X_t$ = past 10 mins, $Y_t$ = next 1 min).

#### 2. The Model (BiLSTM-Attention)
The core innovation is the **Attention Layer**, which allows the model to "focus" on specific past time steps (e.g., the start of a turn) rather than treating all history equally.

```python
# Code Snippet: Custom Attention Mechanism
def attention_3d_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(inputs.shape[1], activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul
    # Permute dimensions to apply dense layer to the time axis
    a = Permute((2, 1))(inputs)
    a = Dense(inputs.shape[1], activation='softmax')(a)
    
    # Re-permute to original orientation
    a_probs = Permute((2, 1), name='attention_vec')(a)
    
    # Multiply weights with original inputs
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul
```
🚀 Usage Instructions
Clone the repository:

Bash

git clone https://github.com/himamuralik/Ship-trajectory-prediction-benchmarking-dissertation.git
cd Ship-trajectory-prediction-benchmarking-dissertation
Install dependencies:

Bash

pip install -r requirements.txt
Run the Notebook: Open ship_trajectory_prediction_final_code.ipynb in Jupyter Notebook or VS Code to run the training pipeline and view the interactive Folium maps.

📈 Visual Results
(Run the notebook to generate these interactive maps)

Blue Line: Actual Ship Path (Ground Truth)

Red Line: Predicted Trajectory (BiLSTM-Attention)

Green Markers: Start/End Points

👨‍💻 Author
Hima Murali Kattur

LinkedIn

GitHub
