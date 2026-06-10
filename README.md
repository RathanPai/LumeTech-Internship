# NILM Project – IAWE & Seq2Point Pipelines

## 📚 Overview
This repository contains two Non‑Intrusive Load Monitoring (NILM) pipelines built around the **IAWE** residential dataset and a **Seq2Point** deep‑learning framework. The goal is to demonstrate high‑quality disaggregation of multiple appliances from mains data.

---

## 📂 Directory Structure
```
/data/Projects/NILM/
├── IAWE/                # IAWE dataset pipeline 
│   ├── Data/            # Raw .h5 file (iawe.h5)
│   ├── NILM_IAWE_Disaggregation.ipynb   # Jupyter notebook (training & evaluation)
│   ├── create_notebook.py  # Python script that generates the notebook
│   └── README.md       # Dataset‑specific documentation
│
├── Seq2point/          # Generic Seq2Point framework
│   ├── generate_notebook.py   # Notebook generator for generic datasets
│   └── config.yaml      # Configuration file
│
└── README.md           # **This file** – project‑wide overview
```

---

## 📊 Dataset Comparison & Selection Justification
When building a NILM pipeline, selecting the right dataset is critical. Here is how the **IAWE** dataset compares to other standard academic datasets, and why it was chosen for this project:

| Dataset | Location | Sampling Rate | Signals Provided | Pros / Cons |
|---------|----------|---------------|------------------|-------------|
| **IAWE** | India | **1 Hz** | $V_{rms}$, $I_{rms}$, Active, Reactive, Apparent Power, Frequency, PF | **Chosen:** Excellent ambient weather context (heavy AC loads), provides full voltage/current metrics, and captures simultaneous heavy dynamic loads (AC + Washing Machine). |
| **REFIT** | UK | 8 seconds | Active Power only | **Con:** Missing $V_{rms}$ and $I_{rms}$, which restricts advanced feature engineering. |
| **REDD** | USA | 1 Hz (Mains), 3 sec (App) | Active Power, Apparent Power | **Con:** Good baseline, but heavily biased towards North American heating systems rather than cooling/dynamic ambient loads. |
| **UK-DALE** | UK | 1 Hz / 16 kHz | Active Power, Apparent Power, raw V/I (subset) | **Con:** Very large and complex to parse; 16kHz raw waveform data is overkill for basic Seq2Point without immense computational resources. |
| **PLAID / BLUED** | USA | 12 kHz - 30 kHz | Raw Voltage & Current Waveforms | **Con:** Excellent for harmonic/FFT analysis, but strictly event-based (PLAID) or computationally expensive for simple week-long time-series deep learning. |

---

## 🗂️ IAWE Dataset Pipeline
### What is IAWE?
- **Source:** Indian dataset collected in New Delhi.
- **Sampling:** Raw sensor data at roughly **1 Hz**.
- **Available Signals:** Voltage ($V_{rms}$), Current ($I_{rms}$), Active Power, Reactive Power, Apparent Power, Frequency, Power Factor.
- **Target Appliances:** Air Conditioner and Washing Machine.

### Preprocessing (IAWE)
1. **Custom HDF5 loader:** Uses the `tables` library to bypass pandas HDFStore string‑encoding issues found in older `.h5` files.
2. **Timestamp handling:** Converts Unix timestamps to timezone‑aware `Asia/Kolkata` datetime.
3. **Resampling:** Aggregates the 1 Hz data to a **uniform 1‑minute grid** (`.resample('1min').mean()`).
4. **Missing‑value imputation:** Forward‑fills (`ffill`) any gaps caused by sensor dropouts.
5. **Standardization:** Each signal (Mains, Washing Machine, AC) is scaled using a `StandardScaler`.
6. **Windowing:** Slices the continuous series into overlapping windows of **99 minutes**.

### Output & Artifacts
- **Model:** A PyTorch model saved as `nilm_multi_target_model.pth`.
- **Metrics:** Outputs Mean Absolute Error (MAE) and Signal Aggregate Error (SAE).
- **Plots:** Generates time-series plots comparing Ground Truth vs. Prediction for both target appliances.

---

## 📈 Seq2Point Framework
The **Seq2Point** module is a generic framework designed to train a 1D Convolutional Neural Network (CNN) to map a sequence (window) of mains power to a single point (the center of the window) of the target appliance power.

### Architecture
- **Input:** A sequence of mains power readings (e.g., shape `(Batch, 1, 99)`).
- **Network:** Multiple 1-D convolutional layers with ReLU activations, followed by fully-connected dense layers.
- **Output:** The predicted power of the target appliance(s) at the exact midpoint of the input sequence.

### Preprocessing (Seq2Point)
1. **Data Alignment:** Ensures the mains and appliance data are aligned on the same timestamps.
2. **Normalization:** Scales the input sequences so the neural network gradients remain stable.
3. **Sequence-to-Point Generation:** A sliding window moves across the data. For every window of length $W$, the label $Y$ is the actual power of the appliance at index $W/2$.

### Output
- Saves the trained model weights to disk.
- Automatically calculates accuracy metrics (MAE, RMSE, F1-score for appliance ON states).
- Visualizes the disaggregation results.

---

## 🚀 How to Run the Project
**Prerequisite:** Ensure the `handwriting_gen` conda environment is active.
```bash
conda activate handwriting_gen
```

**To generate and run the IAWE Notebook:**
```bash
cd /data/Projects/NILM
python IAWE/create_notebook.py
jupyter nbconvert --execute IAWE/NILM_IAWE_Disaggregation.ipynb --to html
```
