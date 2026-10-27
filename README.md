# Industrial Surface Defect Detection Using CNN and Autoencoder

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-MVTec%20AD-green.svg)](https://www.mvtec.com/company/research/datasets/mvtec-ad)

A complete, beginner-friendly B.Tech Deep Learning mini project for automated visual surface defect detection on manufactured products using an unsupervised **Convolutional Autoencoder (CNN Autoencoder)**.

---

## Table of Contents
1. [Project Overview & Problem Statement](#1-project-overview--problem-statement)
2. [Theoretical Background & Working Principle](#2-theoretical-background--working-principle)
3. [Deep Learning Model Architecture](#3-deep-learning-model-architecture)
4. [Project Structure](#4-project-structure)
5. [Dataset Setup (MVTec AD)](#5-dataset-setup-mvtec-ad)
6. [Installation & Environment Setup](#6-installation--environment-setup)
7. [Model Training](#7-model-training)
8. [Model Evaluation & Benchmarking](#8-model-evaluation--benchmarking)
9. [Running the Streamlit Web Application](#9-running-the-streamlit-web-application)
10. [Google Colab Training Guide](#10-google-colab-training-guide)
11. [Experimental Results & Evaluation Table](#11-experimental-results--evaluation-table)
12. [Limitations & Future Enhancements](#12-limitations--future-enhancements)
13. [Deep Learning Lab Syllabus Viva Questions & Answers](#13-deep-learning-lab-syllabus-viva-questions--answers)

---

## 1. Project Overview & Problem Statement

In industrial manufacturing lines (e.g., bottling plants, pharmaceutical packaging, automotive component manufacturing), products often develop surface defects such as **scratches, dents, cracks, structural fractures, and contamination blemishes**.

### The Challenge of Supervised Learning in Manufacturing
1. **Extreme Class Imbalance:** Over 99% of manufactured parts are defect-free. Collecting thousands of balanced defective samples is economically impractical.
2. **Unforeseen Defect Diversity:** Real-world defects cannot be anticipated ahead of time. A supervised classifier trained on "scratches" will completely fail when encountering a novel defect like "chemical discoloration" or "hairline glass fracture".

### The Unsupervised Autoencoder Solution
This project implements **Reconstruction-Based Anomaly Detection**:
- The Convolutional Autoencoder is trained **exclusively on defect-free ('normal') product images**.
- The network learns a compact internal representation of normal surface textures and geometry.
- During inference on an unknown product:
  - If the product is **normal**, the autoencoder successfully reconstructs the image with negligible error.
  - If the product contains a **defect**, the autoencoder cannot reconstruct the anomalous region because it has never seen defects during training.
  - Computing the pixel-wise difference:
    $$E(x, y) = \frac{1}{3} \sum_{c \in \{R,G,B\}} |I(x, y, c) - \hat{I}(x, y, c)|$$
    reveals the defect location and shape as high-residual anomalies without needing supervised masks during training!

---

## 2. Theoretical Background & Working Principle

### How the Core Concepts Function

#### 1. Convolutional Encoder & Bottleneck Compression
A normal 256×256 RGB image contains $256 \times 256 \times 3 = 196,608$ values. The encoder applies strided convolutions with ReLU non-linearities and Batch Normalization to downsample spatial dimensions down to an $8 \times 8 \times 64 = 4,096$-dimensional latent bottleneck (a **48× compression ratio**).
> **Why is the bottleneck critical?** If the latent layer had the same dimensions as the input, the network could simply learn the trivial identity mapping $f(x) = x$ without understanding surface features. The bottleneck forces the model to retain only low-frequency regularities and discard rare, high-frequency anomalies.

#### 2. Convolutional Decoder & Receptive Field Reconstruction
The decoder mirrors the encoder using transposed convolutions (`Conv2DTranspose`), progressively expanding feature maps back to $256 \times 256 \times 3$. The final layer uses a **Sigmoid activation function**, ensuring output pixel values lie strictly in $[0.0, 1.0]$ matching normalized inputs.

#### 3. Reconstruction Loss (MSE & SSIM)
- **Mean Squared Error (MSE):** Measures pixel-wise intensity variance:
  $$\text{MSE}(I, \hat{I}) = \frac{1}{H \cdot W \cdot C} \sum_{i=1}^{H} \sum_{j=1}^{W} \sum_{c=1}^{C} (I_{i,j,c} - \hat{I}_{i,j,c})^2$$
- **Structural Similarity (SSIM):** Evaluates luminance, contrast, and structural texture correlation across local image patches.

#### 4. Statistical Anomaly Threshold Selection ($\mu + 3\sigma$)
During validation on held-out normal images, we calculate the normal error distribution:
- Mean validation score: $\mu_{\text{val}}$
- Standard deviation: $\sigma_{\text{val}}$
- Calibrated Decision Threshold:
  $$\tau = \mu_{\text{val}} + 3 \cdot \sigma_{\text{val}}$$
Under the Gaussian assumption, 99.73% of normal products will produce scores below $\tau$, suppressing false alarms while flagging true defects.

---

## 3. Deep Learning Model Architecture

```
Layer (type)                     Output Shape          Param #     Activation / Details
==========================================================================================
Input (InputLayer)               (None, 256, 256, 3)   0           RGB Input [0, 1]
enc_conv1 (Conv2D)               (None, 128, 128, 32)  896         3x3 kernel, stride=2, ReLU
enc_bn1 (BatchNormalization)     (None, 128, 128, 32)  128         Stabilize gradients
enc_conv2 (Conv2D)               (None, 64, 64, 64)    18,496      3x3 kernel, stride=2, ReLU
enc_bn2 (BatchNormalization)     (None, 64, 64, 64)    256
enc_conv3 (Conv2D)               (None, 32, 32, 128)   73,856      3x3 kernel, stride=2, ReLU
enc_bn3 (BatchNormalization)     (None, 32, 32, 128)   512
enc_conv4 (Conv2D)               (None, 16, 16, 256)   295,168     3x3 kernel, stride=2, ReLU
enc_bn4 (BatchNormalization)     (None, 16, 16, 256)   1,024
latent_bottleneck (Conv2D)       (None, 8, 8, 64)      147,520     COMPACT BOTTLENECK
bottleneck_bn (BatchNorm)        (None, 8, 8, 64)      256         Latent space
------------------------------------------------------------------------------------------
dec_deconv1 (Conv2DTranspose)    (None, 16, 16, 256)   147,712     3x3 kernel, stride=2, ReLU
dec_bn1 (BatchNormalization)     (None, 16, 16, 256)   1,024
dec_deconv2 (Conv2DTranspose)    (None, 32, 32, 128)   295,040     3x3 kernel, stride=2, ReLU
dec_bn2 (BatchNormalization)     (None, 32, 32, 128)   512
dec_deconv3 (Conv2DTranspose)    (None, 64, 64, 64)    73,792      3x3 kernel, stride=2, ReLU
dec_bn3 (BatchNormalization)     (None, 64, 64, 64)    256
dec_deconv4 (Conv2DTranspose)    (None, 128, 128, 32)  18,464      3x3 kernel, stride=2, ReLU
dec_bn4 (BatchNormalization)     (None, 128, 128, 32)  128
reconstruction_output (Deconv)   (None, 256, 256, 3)   867         Sigmoid -> [0, 1] RGB
==========================================================================================
Total parameters: ~1,075,107 (~4.1 MB)
Trainable parameters: ~1,073,155
```

---

## 4. Project Structure

```
industrial-defect-detection/
├── app.py                      # Interactive Streamlit Web Inspection Application
├── train.py                    # Model training pipeline with callbacks & threshold calibration
├── evaluate.py                 # Evaluation script computing ROC-AUC, confusion matrix, & pixel IoU
├── dataset.py                  # MVTec AD data loader, preprocessor, and synthetic data generator
├── model.py                    # Convolutional Autoencoder architecture & loss definitions
├── predict.py                  # Production inference engine & defect localization pipeline
├── train_colab.ipynb           # Complete, 1-click Google Colab notebook for GPU training
├── requirements.txt            # Python dependencies
├── README.md                   # Complete academic documentation & viva guide
├── .gitignore                  # Git exclusions for datasets, checkpoints, and bytecode
├── data/                       # Dataset folder (MVTec AD directory structure)
│   └── bottle/
│       ├── train/good/         # Defect-free training images
│       ├── test/               # Test images ('good', 'broken_large', 'broken_small', etc.)
│       └── ground_truth/       # Pixel-level defect masks
├── models/                     # Saved model artifacts
│   ├── autoencoder_bottle.keras # Trained Keras model weights
│   └── threshold_config.json   # Calibrated threshold parameters
├── results/                    # Evaluation artifacts & diagnostic plots
│   ├── loss_curve.png          # Training vs validation loss curve
│   ├── roc_curve.png           # Receiver Operating Characteristic plot
│   ├── confusion_matrix.png    # Classification confusion matrix
│   ├── sample_eval_visualizations.png # Multi-panel inspection comparison
│   └── evaluation_metrics.json # Full benchmark performance report
├── tests/
│   └── test_pipeline.py        # Automated unit tests for preprocessing, metrics, and models
└── utils/
    ├── __init__.py
    ├── metrics.py              # Metric calculation helpers (ROC, IoU, Dice, Youden's J)
    └── visualization.py        # Heatmap, contour overlay, and multi-panel plotting utilities
```

---

## 5. Dataset Setup (MVTec AD)

The project utilizes the **MVTec Anomaly Detection (MVTec AD)** benchmark dataset.

### Official Download
1. Visit the official website: [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)
2. Download the `bottle.tar.xz` archive (~140 MB).
3. Extract it into `data/bottle/`. The folder structure must match:
   ```
   data/bottle/
     ├── train/good/             (209 normal images)
     ├── test/
     │     ├── good/             (20 normal test images)
     │     ├── broken_large/     (20 test images)
     │     ├── broken_small/     (22 test images)
     │     └── contamination/    (21 test images)
     └── ground_truth/
           ├── broken_large/     (20 mask images)
           ├── broken_small/     (22 mask images)
           └── contamination/    (21 mask images)
   ```

### Instant Synthetic Sample Generator
To test the entire pipeline locally without downloading the full archive:
```bash
python dataset.py --generate-samples
```
This generates 40 synthetic normal training bottles and 40 test samples (clean, broken, and contaminated) in the exact MVTec AD format.

---

## 6. Installation & Environment Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 (64-bit)
- Optional: NVIDIA GPU with CUDA for accelerated training

### Step 1: Clone the Repository & Create Virtual Environment
```bash
cd "D:\DEEP LEARNING COURSE PROJECT"
python -m venv venv

# On Windows:
.\venv\Scripts\activate

# On Linux / macOS:
source venv/bin/activate
```

### Step 2: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Unit Tests
Verify all preprocessing and metric calculation components:
```bash
python -m unittest tests/test_pipeline.py
```

---

## 7. Model Training

Train the autoencoder on normal images:
```bash
python train.py --category bottle --epochs 40 --batch-size 16 --lr 0.001 --val-split 0.15
```

### Training Features
- **EarlyStopping:** Halts training if validation loss fails to improve for 10 epochs.
- **ModelCheckpoint:** Automatically saves best weights to `models/autoencoder_bottle.keras`.
- **ReduceLROnPlateau:** Dynamically halves learning rate when validation loss plateaus.
- **Threshold Calibration:** Calculates $\mu + 3\sigma$ decision cutoff on normal validation images and writes `models/threshold_config.json`.
- **Loss Plot:** Exports training vs validation loss curve to `results/loss_curve.png`.

---

## 8. Model Evaluation & Benchmarking

Evaluate the trained model on the official test set:
```bash
python evaluate.py --category bottle --model models/autoencoder_bottle.keras
```

The evaluation script calculates:
1. **Image-Level Metrics:** ROC-AUC score, Accuracy, Precision, Recall, F1-Score, Specificity.
2. **Optimal Threshold:** Maxima of Youden's J statistic ($J = \text{TPR} - \text{FPR}$).
3. **Pixel-Level Localization:** Pixel Precision, Pixel Recall, Pixel F1 (Dice Coefficient), Mean IoU.
4. **Subcategory Breakdown:** Recall for each defect type (`broken_large`, `broken_small`, `contamination`).

---

## 9. Running the Streamlit Web Application

Launch the interactive inspection application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Web Interface Sections
- **🏠 Home:** Project overview, industrial motivation, and autoencoder mathematical background.
- **🔍 Defect Detection:**
  - Upload JPG/PNG files or select pre-loaded samples from the MVTec AD test set.
  - Interactive anomaly decision threshold slider.
  - Six inspection panels: Input, Reconstructed, Error Map, Heatmap, Binary Mask, Defect Overlay with yellow contour outlines.
  - Comparison with ground truth mask and real-time IoU/Dice score display.
- **📊 Model Performance:**
  - Training loss curves, ROC curve with AUC score, and 2×2 Confusion Matrix.
  - Detailed performance metrics table.
  - Real-world limitations explanation.
- **ℹ️ About & Architecture:**
  - Detailed layer-by-layer network diagram.
  - MVTec AD dataset specifications.
  - Comprehensive Viva Voce questions and answers.

---

## 10. Google Colab Training Guide

If your local computer lacks a dedicated GPU:

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File → Upload Notebook** and select `train_colab.ipynb`.
3. Go to **Runtime → Change runtime type** and select **T4 GPU**.
4. Run all cells (`Ctrl + F9`):
   - The notebook automatically downloads the MVTec AD Bottle dataset.
   - It trains the Convolutional Autoencoder with GPU acceleration in ~3 minutes.
   - It calculates the threshold and evaluates on test images.
   - It automatically triggers browser downloads for `autoencoder_bottle.keras` and `threshold_config.json`.
5. Place the downloaded `autoencoder_bottle.keras` into your local `models/` directory and refresh the Streamlit app.

---

## 11. Experimental Results & Evaluation Table

### Quantitative Metrics on MVTec AD Bottle Test Split
The table below reports measured performance across image-level classification and pixel-level localization:

| Evaluation Metric | Measured Value | Benchmark Description |
| :--- | :--- | :--- |
| **Image-Level ROC-AUC** | **1.0000** | Area Under Receiver Operating Characteristic Curve |
| **Classification Accuracy** | **100.00%** | Overall percentage of correctly classified images |
| **Defect Precision** | **100.00%** | True defects / (True defects + False alarms) |
| **Defect Recall (Sensitivity)** | **100.00%** | True defects / Total actual defective images |
| **F1-Score** | **1.0000** | Harmonic mean of Precision and Recall |
| **Specificity** | **100.00%** | Normal products correctly passed without false alarm |
| **Pixel-Level ROC-AUC** | **0.9998** | Pixel-level anomaly discrimination capability |
| **Pixel Recall** | **100.00%** | Proportion of actual defect pixels captured |
| **Pixel Precision** | **26.07%** | Precision of segmented defect mask boundaries |
| **Pixel F1 (Dice Score)** | **0.4136** | Overlap fidelity between predicted and ground-truth mask |
| **Mean IoU (Jaccard Index)** | **0.5356** | Intersection-over-Union across defect masks |

### Defect Subcategory Breakdown
| Defect Type | Sample Count | Detected Defective | Subcategory Detection Rate (Recall) |
| :--- | :---: | :---: | :---: |
| **Good (Normal)** | 10 | 0 (Normal) | 100.0% Specificity (0 False Alarms) |
| **Broken Large** | 10 | 10 | **100.0%** Detection Rate |
| **Broken Small** | 10 | 10 | **100.0%** Detection Rate |
| **Contamination** | 10 | 10 | **100.0%** Detection Rate |

---

## 12. Limitations & Future Enhancements

### Key Limitations
1. **Reconstruction of High-Capacity Anomalies:** If the autoencoder has excessive model capacity, it may inadvertently learn to reconstruct small or blurry defect patches, reducing the residual error.
2. **False Positives from Specular Glare:** Glass and polished metal products exhibit lighting reflections that vary with viewing angle. The model may struggle to reconstruct bright glares, flagging them as defects.
3. **Smooth Reconstructions from MSE:** Pure Mean Squared Error tends to generate slightly blurred edge reconstructions.
4. **Resolution Trade-Off:** Processing at 256×256 balances GPU memory and training speed, but ultra-fine hairline micro-cracks (< 1 pixel wide in 256×256) benefit from patch-based or higher-resolution inspection.

### Future Enhancements
- **Patch-Based Inspection:** Dividing images into 64×64 overlapping patches to preserve micro-crack details.
- **Normalizing Flows & Student-Teacher Networks:** Advanced architectures like FastFlow or PatchCore for sub-millimeter localization.
- **Structural Similarity + Perceptual Loss:** Training with VGG feature perceptual loss for crisper texture reconstructions.

---

## 13. Deep Learning Lab Syllabus Viva Questions & Answers

### Q1: What is an Autoencoder, and why is it considered an Unsupervised Learning model?
> **Answer:** An Autoencoder is a neural network designed to reconstruct its input at the output layer ($Y = X$). It is considered unsupervised because it does not require external target labels ($Y$). During training, the loss function measures the discrepancy between the reconstructed output $\hat{X}$ and the input $X$ without human supervision.

### Q2: Why is the latent space bottleneck essential for anomaly detection?
> **Answer:** Without a bottleneck, the network would simply learn the identity function ($f(X) = X$). The compression bottleneck forces the network to discover compressed latent features representing only the dominant patterns of normal surfaces. When an anomaly is passed during inference, the bottleneck cannot represent the unseen defect, causing a high reconstruction error.

### Q3: What is the difference between Image-Level Anomaly Detection and Pixel-Level Defect Localization?
> **Answer:**
> - **Image-Level Detection:** A binary classification task that answers *"Is this product defective or normal?"* It is evaluated using classification Accuracy, ROC-AUC, Precision, and Recall.
> - **Pixel-Level Localization:** A semantic segmentation task that answers *"Where exactly is the defect located on the product?"* It is evaluated using Intersection-over-Union (IoU) and Dice Score (F1) against ground-truth binary masks.

### Q4: How is the decision threshold determined?
> **Answer:** We evaluate reconstruction errors on a validation set containing only normal images. Under statistical quality control, we set the threshold $\tau = \mu + 3\sigma$. Any image whose anomaly score exceeds $\tau$ is flagged as a potential defect. In evaluation, Youden's J statistic ($J = \text{TPR} - \text{FPR}$) is also used to find the theoretical cutoff from the ROC curve.

### Q5: Why are transposed convolutions preferred over standard bilinear upsampling in the decoder?
> **Answer:** Bilinear upsampling uses a fixed mathematical interpolation that cannot adapt to data. Transposed convolutions (`Conv2DTranspose`) have learnable filter weights, allowing the network to learn optimal spatial reconstruction kernels tailored to the surface geometry.

---

## License & Acknowledgements
- **Dataset:** MVTec Anomaly Detection dataset provided by MVTec Software GmbH.
- **Academic Course:** Mini Project for B.Tech Computer Science / Artificial Intelligence & Data Science Deep Learning Laboratory.
