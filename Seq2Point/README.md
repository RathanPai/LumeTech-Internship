# Multi-Target NILM using Seq2Point

This repository contains the complete pipeline for Non-Intrusive Load Monitoring (NILM) focusing on disaggregating two target appliances (Washing Machine and Kettle) simultaneously using a speed-optimized Deep Learning Sequence-to-Point (Seq2Point) 1D Convolutional Neural Network.

---

## 1. Preprocessing & Model Parameters

The notebook `nilm_seq2point.ipynb` implements the following preprocessing pipeline and model parameters:

* **Sampling Rate**: 8-seconds. The raw data is resampled to 8-second intervals to maintain consistency and handle missing values via forward filling. This provides a dense enough signal for the Seq2Point architecture to capture the high-frequency state transitions (ON/OFF cycles) of appliances perfectly without leading to excessively long sequences.
* **Window Size**: 599 time steps (equivalent to approximately 80 minutes).
* **Features Used/Extracted**: Aggregate (Mains) active power. The input features are 599-step sequences of the aggregate power, and the targets are the active power of the Washing Machine and the Kettle at the exact *center* of the window (Seq2Point architecture).
* **Model Used**: Multi-Target Sequence-to-Point 1D Convolutional Neural Network (CNN). The network utilizes a shared convolutional feature extractor and outputs two predictions simultaneously.

### A. Speed Optimizations
The pipeline includes several advanced optimizations for faster training:
- **Training Stride**: A stride of 10 was applied during training dataset generation. This avoids processing heavily redundant overlapping sequences and significantly speeds up iteration time while preserving accuracy.
- **Mixed Precision**: Automatic Mixed Precision (AMP) was implemented during the PyTorch training loop to utilize GPU Tensor Cores, yielding 1.5x - 2x faster training times.
- **Data Loading**: Increased batch sizes and multiprocessing workers were utilized to maximize GPU utilization.

---

## 2. Results Obtained

The Multi-Target Seq2Point model was evaluated on the test dataset (House 20). The following evaluation metrics were obtained:

* **MAE - Washing Machine**: 6.97 W (which is ~1.83% of the appliance's average operating power when ON)
* **MAE - Kettle**: 6.86 W (which is ~0.25% of the appliance's average operating power when ON)
* **RMSE - Washing Machine**: 68.24 W
* **RMSE - Kettle**: 78.86 W
* **F1 Score - Washing Machine**: 0.645 (for power > 20W)
* **F1 Score - Kettle**: 0.856 (for power > 2000W)

*(Note: MAE stands for Mean Absolute Error and RMSE stands for Root Mean Squared Error)*
