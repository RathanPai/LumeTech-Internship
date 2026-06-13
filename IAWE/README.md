# Multi-Target NILM on the IAWE Dataset

This repository contains the complete pipeline for Non-Intrusive Load Monitoring (NILM) on the IAWE dataset, focusing on disaggregating two target appliances (Washing Machine and Air Conditioner) simultaneously using a Deep Learning Convolutional Neural Network (Sequence-to-Point).

---

## 1. Dataset Justification: Why IAWE?

When deciding on a dataset for this NILM project, **IAWE (Indian Dataset for Ambient Water and Energy)** was chosen over other popular datasets like **REDD** (USA) or **REFIT** (UK) for the following reasons:

1. **Regional Context & Ambient Climate:** Power consumption patterns are heavily influenced by climate. The IAWE dataset, collected in New Delhi, captures heavy Air Conditioner usage alongside significant dynamic loads like Washing Machines. This provides a uniquely challenging and realistic scenario compared to UK/US datasets where heating might dominate over cooling.
2. **High-Frequency Ground Truth:** The raw data in IAWE is captured at **1Hz** (1 sample per second). This rich granularity captures detailed transient signatures of appliance state-changes, providing a solid foundation before resampling.
3. **Simultaneous Usage:** It contains clear, overlapping usage of major appliances, making it a perfect candidate to demonstrate a **Multi-Target** Deep Learning model where the network must disentangle simultaneous signals.

---

## 2. Preprocessing & Model Parameters

Deep Learning models cannot process raw time-series data without careful preparation. The notebook `NILM_IAWE_Disaggregation.ipynb` implements the following preprocessing pipeline and model parameters:

* **Sampling Rate**: 1-minute (resampled from 1Hz raw data by taking the mean power over each minute).
* **Window Size**: 99 minutes.
* **Features Used/Extracted**: Active Power (Index 0) from the Mains. The input features are 99-minute sequences of Mains Active Power, and the targets are the Active Power of the Washing Machine and the AC at the exact *center* of the window (Seq2Point architecture).

### A. Custom HDF5 Parsing
Older HDF5 files (`iawe.h5`) often trigger a `TypeError` in modern Python 3 `pandas` versions due to byte/string encoding mismatches in the column headers. To build a robust pipeline, a custom loader was written using the `tables` library. It reads the raw data arrays and reconstructs the `pandas` DateTimeIndex natively, ensuring no environment downgrades were needed.

### B. Missing Value Imputation
Sensors occasionally drop packets. We used **Forward Fill (`ffill`)** to handle missing data. This assumes that if a sensor drops a reading, the appliance remained in its last known power state until the next valid reading.

### C. Normalization
All power data was standardized using a **StandardScaler** (zero mean, unit variance). Neural Networks learn much faster and avoid vanishing/exploding gradients when input features are scaled, rather than feeding raw values that fluctuate between 0W and 3000W.

---

## 3. Results Obtained

The Multi-Target Sequence-to-Point 1D CNN model was evaluated on the test dataset. The following evaluation metrics were obtained:

* **MAE - Washing Machine**: 173.21 (which is ~82.32% of the appliance's average operating power when ON) (High due to higher sampling of the dataset)
* **MAE - AC**: 49.54 (which is ~2.98% of the appliance's average operating power when ON)
* **SAE - Washing Machine**: 0.4777
* **SAE - AC**: 0.0029

---
