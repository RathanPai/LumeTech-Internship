# Multi-Target NILM on the IAWE Dataset

This repository contains the complete pipeline for Non-Intrusive Load Monitoring (NILM) on the IAWE dataset, focusing on disaggregating two target appliances (Washing Machine and Air Conditioner) simultaneously using a Deep Learning Convolutional Neural Network (Sequence-to-Point).

---

## 1. Dataset Justification: Why IAWE?

When deciding on a dataset for this NILM project, **IAWE (Indian Dataset for Ambient Water and Energy)** was chosen over other popular datasets like **REDD** (USA) or **REFIT** (UK) for the following reasons:

1. **Regional Context & Ambient Climate:** Power consumption patterns are heavily influenced by climate. The IAWE dataset, collected in New Delhi, captures heavy Air Conditioner usage alongside significant dynamic loads like Washing Machines. This provides a uniquely challenging and realistic scenario compared to UK/US datasets where heating might dominate over cooling.
2. **High-Frequency Ground Truth:** The raw data in IAWE is captured at **1Hz** (1 sample per second). This rich granularity captures detailed transient signatures of appliance state-changes, providing a solid foundation before resampling.
3. **Simultaneous Usage:** It contains clear, overlapping usage of major appliances, making it a perfect candidate to demonstrate a **Multi-Target** Deep Learning model where the network must disentangle simultaneous signals.

---

## 2. Preprocessing Explained

Deep Learning models cannot process raw time-series data without careful preparation. The notebook `NILM_IAWE_Disaggregation.ipynb` implements the following neat preprocessing pipeline:

### A. Custom HDF5 Parsing
Older HDF5 files (`iawe.h5`) often trigger a `TypeError` in modern Python 3 `pandas` versions due to byte/string encoding mismatches in the column headers. To build a robust pipeline, a custom loader was written using the `tables` library. It reads the raw data arrays and reconstructs the `pandas` DateTimeIndex natively, ensuring no environment downgrades were needed.

### B. Resampling & Alignment
The raw 1Hz data was resampled to a **1-minute resolution (`1T`)** by taking the mean power over each minute. 
- *Why?* Processing 73 days of 1-second data is computationally expensive. 1-minute intervals perfectly balance dataset size with the required resolution to detect multi-minute appliance cycles (like a Fridge compressor turning on/off).

### C. Missing Value Imputation
Sensors occasionally drop packets. We used **Forward Fill (`ffill`)** to handle missing data. This assumes that if a sensor drops a reading, the appliance remained in its last known power state until the next valid reading.

### D. Normalization
All power data was standardized using a **StandardScaler** (zero mean, unit variance). Neural Networks learn much faster and avoid vanishing/exploding gradients when input features are scaled, rather than feeding raw values that fluctuate between 0W and 3000W.

### E. Windowing (Sequence-to-Point)
We transformed the time-series into overlapping sliding windows (e.g., a window size of 99 minutes).
- **Input (`X`)**: A 99-minute sequence of the Mains power.
- **Target (`Y`)**: The actual power of the Washing Machine and the AC at the exact *center* of that 99-minute window.

---

## 3. 📝 Presentation Notes (For Q&A)

*If you are presenting this work, here are anticipated questions regarding the preprocessing and design choices:*

**Q1: Why resample to 1-minute instead of keeping the 1Hz data?**
> **Answer:** Training a CNN on 73 days of 1Hz data would result in sequences that are too long for the network's receptive field to process efficiently, leading to vanishing gradients. 1-minute data captures the macroscopic state transitions (ON/OFF cycles) of both the AC and Washing Machine perfectly, while drastically reducing training time for rapid experimentation.

**Q2: Why use Forward-Fill for missing data instead of Interpolation?**
> **Answer:** Linear interpolation can artificially "hallucinate" gradual power ramps during a sensor gap, which is physically inaccurate for appliances that just snap ON or OFF. Forward-filling assumes the appliance stayed in its last known state, which is the safest physical assumption for short sensor dropouts.

**Q3: Why build a Single Multi-Target model instead of two separate models?**
> **Answer:** A Multi-Target model shares the initial Convolutional feature extractors. The network learns the general background "Mains" noise once, and uses those shared features to predict both appliances simultaneously. This cuts the number of parameters in half, speeds up inference, and theoretically allows the model to learn negative correlations (e.g., if the model detects an AC signature, it helps rule out the Washing Machine for that specific power draw).

**Q4: How did you bypass the pandas HDFStore read error without downgrading the environment?**
> **Answer:** We accessed the HDF5 file at a lower level using the `tables` library. We directly extracted the `index` (Unix timestamps) and `values_block_0` (Active Power) arrays, and manually reconstructed the `pandas.Series` using `pd.to_datetime`. This demonstrates strong data engineering skills beyond just calling standard pandas functions.
