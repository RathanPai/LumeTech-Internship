# Non-Intrusive Load Monitoring (NILM) Projects

This repository contains a collection of deep learning-based Non-Intrusive Load Monitoring (NILM) projects. NILM, or energy disaggregation, is the task of deducing the power consumption of individual appliances from a single aggregate master meter reading. 

This repository is divided into three distinct sub-projects, each tackling energy disaggregation with different datasets, architectures, and objectives.

---

## 1. Multi-Appliance NILM Disaggregator (PLAID Dataset)
**Directory:** [`PLAIDD`](./PLAIDD)

This project implements a state-of-the-art disaggregator using high-frequency data to separate six simultaneous appliances (Microwave, Heater, Fridge, Air Conditioner, Washing Machine, and Vacuum).

* **Features:** Extracts perfectly linear, additive 247-dimensional features using Fast Fourier Transforms (FFT) for the first 60 harmonics (up to 3600Hz) from 30kHz high-frequency data.
* **Architecture (`NNAN_Model`):** A custom hybrid network featuring 1D CNNs, LSTMs, and Attention mechanisms designed to capture high-level spectral patterns and temporal dynamics.
* **Key Innovations:** Dynamic target scaling with Huber loss, realistic sparse data synthesis, and automated hyperparameter optimization using Optuna.

## 2. Multi-Target NILM on the IAWE Dataset (Experimentational)
**Directory:** [`IAWE`](./IAWE)

A pipeline focusing on disaggregating heavy climate-influenced loads, specifically Washing Machines and Air Conditioners, using the Indian Dataset for Ambient Water and Energy (IAWE).

* **Dataset Context:** Utilizes the IAWE dataset for its 1Hz granularity and real-world simultaneous usage of major cooling and dynamic loads.
* **Architecture:** Multi-Target Sequence-to-Point (Seq2Point) 1D Convolutional Neural Network.
* **Data Engineering:** Features a custom HDF5 parser for robust loading, 1-minute resampling over a 99-minute window, and rigorous data normalization for improved model convergence.

## 3. Speed-Optimized Multi-Target Seq2Point
**Directory:** [`Seq2Point`](./Seq2Point)

A highly optimized Multi-Target Sequence-to-Point (Seq2Point) CNN focusing on disaggregating Washing Machines and Kettles simultaneously, optimized for faster iteration and training times.

* **Architecture:** Seq2Point 1D CNN processing 8-second resolution data over an 80-minute window (599 time steps).
* **Speed Optimizations:** 
  * Implements a training stride of 10 to avoid redundant processing.
  * Utilizes PyTorch Automatic Mixed Precision (AMP) and Tensor Cores for a 1.5x - 2x speedup.
  * Employs increased batch sizing and multiprocessing for maximized GPU utilization.
* **Results:** Achieves exceptionally low Mean Absolute Error (MAE) and high F1 scores.

---

## Getting Started

Each project is self-contained within its respective directory. Please navigate to the individual directories and refer to their specific `README.md` files for detailed setup instructions, architecture descriptions, and evaluation metrics.
