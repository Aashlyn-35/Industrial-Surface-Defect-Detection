"""
app.py - Streamlit Web Application for Industrial Surface Defect Detection System.

Features:
- Home: Theoretical overview, objectives, and autoencoder workflow.
- Defect Detection: Image upload, sample image picker, real-time autoencoder reconstruction,
  pixel-wise error maps, colorized anomaly heatmaps, threshold adjustments, and contour overlays.
- Model Performance: Loss curves, ROC curves, confusion matrices, and benchmark metrics.
- About: Dataset details, CNN architecture specifications, limitations, and future improvements.
"""

import os
import glob
import json
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import streamlit as st

# Set page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Industrial Surface Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for academic aesthetic
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .status-normal {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 12px 18px;
        border-radius: 6px;
        color: #065F46;
        font-weight: 600;
        font-size: 1.2rem;
    }
    .status-defect {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 12px 18px;
        border-radius: 6px;
        color: #991B1B;
        font-weight: 600;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

MODEL_FILE = "models/autoencoder_bottle.keras"
CONFIG_FILE = "models/threshold_config.json"
RESULTS_DIR = "results"
DATA_DIR = "data"


@st.cache_resource(show_spinner="Loading trained Autoencoder model...")
def load_detector():
    """Cached loader for DefectDetector to prevent reloading model on every interaction."""
    from predict import DefectDetector
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        detector = DefectDetector(model_path=MODEL_FILE, config_path=CONFIG_FILE)
        return detector if detector.is_model_loaded() else None
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None


def get_available_sample_images(category="bottle"):
    """Scans data directory for sample test images across categories."""
    base_path = Path(DATA_DIR) / category / "test"
    samples = {}
    if base_path.exists():
        for defect_folder in sorted(base_path.iterdir()):
            if defect_folder.is_dir():
                imgs = sorted(list(defect_folder.glob("*.png")) + list(defect_folder.glob("*.jpg")))
                if imgs:
                    samples[defect_folder.name] = imgs
    return samples


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
st.sidebar.title("Defect Inspector")
st.sidebar.markdown("**B.Tech Deep Learning Mini Project**")

menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔍 Defect Detection", "📊 Model Performance", "ℹ️ About & Architecture"],
    index=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Settings")

# Check if model is loaded
detector = load_detector()
if detector:
    st.sidebar.success("🟢 Model Loaded: `autoencoder_bottle.keras`")
    default_thresh = detector.threshold
else:
    st.sidebar.warning("⚠️ Model not found locally")
    default_thresh = 0.035

# User adjustable sensitivity slider
active_threshold = st.sidebar.slider(
    "Anomaly Decision Threshold",
    min_value=0.010,
    max_value=0.150,
    value=float(default_thresh),
    step=0.005,
    format="%.4f",
    help="Images with anomaly score above this value are flagged as defective."
)

overlay_opacity = st.sidebar.slider(
    "Defect Overlay Opacity",
    min_value=0.1,
    max_value=0.9,
    value=0.45,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: MVTec AD (Bottle Category)")
st.sidebar.caption("Framework: TensorFlow / Keras")


# ==========================================
# PAGE 1: HOME
# ==========================================
if menu == "🏠 Home":
    st.markdown('<div class="main-title">Industrial Surface Defect Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Unsupervised Anomaly Detection & Defect Localization using Convolutional Autoencoders</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        ### 1. Problem Statement
        In modern high-speed industrial manufacturing (e.g., pharmaceuticals, bottling plants, semiconductor fabrication),
        manual visual inspection of products for scratches, cracks, dents, and contamination is slow, labor-intensive, and prone to fatigue.
        
        Traditional supervised classification models require thousands of labeled defect examples. However, in real factories:
        - Defective items are rare (< 1% of total production).
        - New, unseen defect varieties frequently arise that were not present in the training set.
        
        ### 2. The Unsupervised Solution: Autoencoder Reconstruction
        This project implements an **Unsupervised Convolutional Autoencoder** trained exclusively on defect-free ('normal') products.
        
        #### Core Working Principle:
        1. **Bottleneck Compression (Encoder):** The network compresses high-dimensional 256×256 inspection images through deep convolutional layers down to a compact 8×8×64 latent bottleneck.
        2. **Surface Reconstruction (Decoder):** Transposed convolutions reconstruct the original normal surface patterns.
        3. **Anomaly Flagging (Residual Error):** Because the autoencoder has *never seen defects*, it cannot reconstruct anomalous regions like cracks or spots. The pixel-wise difference between input and reconstruction reveals the exact location and shape of defects!
        """)

    with col2:
        st.info("""
        #### 🎯 Key Project Objectives
        - **Binary Classification:** Classify surface as Normal or Defective.
        - **Defect Localization:** Pinpoint exact pixel coordinates without supervised annotations.
        - **Anomaly Heatmap:** Color-code severity of surface irregularities using the Jet colormap.
        - **Academic Demonstration:** Showcase key DL concepts (CNN, Bottleneck representation, MSE/SSIM loss, ROC-AUC).
        """)

        st.markdown("""
        #### ⚙️ Inspection Workflow
        ```mermaid
        graph TD
            A[Input Product Image] --> B[CNN Encoder]
            B --> C[Latent Bottleneck]
            C --> D[CNN Decoder]
            D --> E[Reconstructed Image]
            A --> F[Residual Error Map]
            E --> F
            F --> G[Gaussian Smoothing]
            G --> H[Thresholding]
            H --> I[Defect Mask & Heatmap]
            H --> J[Normal / Defective Decision]
        ```
        """)


# ==========================================
# PAGE 2: DEFECT DETECTION
# ==========================================
elif menu == "🔍 Defect Detection":
    st.markdown('<div class="main-title">Live Defect Inspection & Localization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload a manufactured product image or pick a test sample to inspect for surface defects.</div>', unsafe_allow_html=True)

    if detector is None:
        st.error("""
        ### ⚠️ Trained Model Not Found!
        The application could not find `models/autoencoder_bottle.keras`.
        
        **How to proceed:**
        1. Train the model locally:
           ```bash
           python train.py --epochs 30 --category bottle
           ```
        2. Or train with GPU acceleration on Google Colab using `train_colab.ipynb` and copy `autoencoder_bottle.keras` into the `models/` folder.
        3. Or click below to automatically generate synthetic data and train a fast demo model for testing:
        """)
        if st.button("🚀 Generate Synthetic Dataset & Train Quick Demo Model"):
            with st.spinner("Generating sample dataset and training demo autoencoder (10 epochs)..."):
                from dataset import generate_synthetic_mvtec_dataset
                from train import train_autoencoder
                generate_synthetic_mvtec_dataset(target_dir=DATA_DIR, category="bottle", num_train=30, num_test_per_type=6)
                train_autoencoder(data_root=DATA_DIR, category="bottle", epochs=10, batch_size=8, output_dir="models", results_dir=RESULTS_DIR)
                st.success("Demo model successfully trained! Please refresh or re-select this tab.")
                st.rerun()

    # Input Selection Tabs
    input_tab1, input_tab2 = st.tabs(["📁 Upload Image File", "🖼️ Choose Sample from Dataset"])
    selected_image = None
    ground_truth_mask = None
    sample_type_label = "Custom Upload"

    with input_tab1:
        uploaded_file = st.file_uploader("Upload an inspection image (JPG, JPEG, PNG)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            try:
                selected_image = Image.open(uploaded_file).convert("RGB")
                sample_type_label = f"Uploaded: {uploaded_file.name}"
            except Exception as e:
                st.error(f"Invalid image file: {e}")

    with input_tab2:
        samples = get_available_sample_images("bottle")
        if samples:
            sample_category = st.selectbox("Select Defect Category", list(samples.keys()))
            sample_files = samples[sample_category]
            sample_file_names = [f.name for f in sample_files]
            chosen_file_name = st.selectbox("Select Image Sample", sample_file_names)
            chosen_path = next(f for f in sample_files if f.name == chosen_file_name)
            selected_image = Image.open(chosen_path).convert("RGB")
            sample_type_label = f"Dataset Sample: {sample_category} / {chosen_file_name}"

            # Check for matching ground truth mask
            gt_path = Path(DATA_DIR) / "bottle" / "ground_truth" / sample_category / f"{chosen_path.stem}_mask.png"
            if gt_path.exists():
                from dataset import load_and_preprocess_mask
                ground_truth_mask = load_and_preprocess_mask(str(gt_path), (256, 256))
        else:
            st.info("No sample dataset images found in `data/bottle/test/`. You can upload an image above, or run `python dataset.py --generate-samples` to create test samples.")

    # Execution if image is selected
    if selected_image is not None and detector is not None:
        st.markdown("---")

        with st.spinner("Processing image through Deep Autoencoder..."):
            result = detector.detect(
                image_input=selected_image,
                custom_threshold=active_threshold
            )

        # Status Banner and Metric Highlights
        is_defect = result["is_defective"]
        score = result["anomaly_score"]
        thresh = result["threshold"]
        defect_area = result["defect_area_pct"]

        if is_defect:
            st.markdown(
                f'<div class="status-defect">🚨 STATUS: POTENTIAL DEFECT DETECTED '
                f'(Score: {score:.4f} &ge; Threshold: {thresh:.4f})</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="status-normal">✅ STATUS: NORMAL PRODUCT SURFACE '
                f'(Score: {score:.4f} &lt; Threshold: {thresh:.4f})</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Anomaly Score", f"{score:.4f}", delta=f"{score - thresh:+.4f} vs threshold", delta_color="inverse")
        with m_col2:
            st.metric("Decision Threshold", f"{thresh:.4f}")
        with m_col3:
            st.metric("Defect Area", f"{defect_area:.2f}%")
        with m_col4:
            st.metric("Sample Source", sample_type_label.split(":")[0])

        st.markdown("### 🔬 Multi-Stage Inspection Breakdown")

        # Visual comparison panels
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.image(result["original_image"], caption="1. Original Input Image", use_container_width=True)
            st.caption("Standardized to 256×256 RGB")
        with col_b:
            st.image(result["reconstructed_image"], caption="2. Autoencoder Reconstruction", use_container_width=True)
            st.caption("Reconstructed normal surface expectation")
        with col_c:
            # Display normalized error map
            err_norm = (np.clip(result["error_map"] / (thresh * 1.5), 0, 1) * 255).astype(np.uint8)
            st.image(err_norm, caption="3. Absolute Reconstruction Error", use_container_width=True)
            st.caption("Pixel-wise intensity difference |Input - Recon|")

        col_d, col_e, col_f = st.columns(3)
        with col_d:
            st.image(result["heatmap"], caption="4. Color Anomaly Heatmap", use_container_width=True)
            st.caption("Jet colormap showing anomaly intensity")
        with col_e:
            st.image(result["mask"], caption="5. Binary Defect Mask", use_container_width=True)
            st.caption(f"Segmented defect pixels (Area: {defect_area:.2f}%)")
        with col_f:
            st.image(result["overlay_image"], caption="6. Defect Overlay & Boundaries", use_container_width=True)
            st.caption("Original surface with highlighted defect perimeter")

        # Ground truth comparison if available
        if ground_truth_mask is not None:
            st.markdown("### 🎯 Ground Truth Comparison")
            gt_col1, gt_col2, gt_col3 = st.columns(3)
            with gt_col1:
                st.image((ground_truth_mask * 255).astype(np.uint8), caption="Ground Truth Annotation", use_container_width=True)
            with gt_col2:
                st.image(result["mask"], caption="Autoencoder Predicted Mask", use_container_width=True)
            with gt_col3:
                # Calculate single image IoU and Dice
                gt_bin = ground_truth_mask > 0
                pred_bin = result["mask"] > 0
                intersection = np.logical_and(gt_bin, pred_bin).sum()
                union = np.logical_or(gt_bin, pred_bin).sum()
                iou = float(intersection) / float(union) if union > 0 else 1.0
                dice = (2.0 * intersection) / (gt_bin.sum() + pred_bin.sum()) if (gt_bin.sum() + pred_bin.sum()) > 0 else 1.0

                st.markdown("#### Localization Accuracy")
                st.metric("Intersection-over-Union (IoU)", f"{iou:.4f}")
                st.metric("Dice Score (F1)", f"{dice:.4f}")


# ==========================================
# PAGE 3: MODEL PERFORMANCE
# ==========================================
elif menu == "📊 Model Performance":
    st.markdown('<div class="main-title">Model Evaluation & Benchmark Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Quantitative metrics and validation curves on the official MVTec AD test split.</div>', unsafe_allow_html=True)

    metrics_json_path = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
    has_metrics = os.path.exists(metrics_json_path)

    if has_metrics:
        with open(metrics_json_path, "r") as f:
            metrics_data = json.load(f)

        img_m = metrics_data.get("image_level_metrics", {})
        pix_m = metrics_data.get("pixel_level_metrics", {})
        sub_m = metrics_data.get("subcategory_breakdown", {})

        st.markdown("### 📈 Measured Benchmark Metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("ROC-AUC Score", f"{img_m.get('roc_auc', 0.0):.4f}")
        with c2:
            st.metric("Accuracy", f"{img_m.get('accuracy', 0.0) * 100:.1f}%")
        with c3:
            st.metric("Precision", f"{img_m.get('precision', 0.0) * 100:.1f}%")
        with c4:
            st.metric("Recall (Sensitivity)", f"{img_m.get('recall', 0.0) * 100:.1f}%")
        with c5:
            st.metric("F1-Score", f"{img_m.get('f1_score', 0.0):.4f}")

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric("Pixel ROC-AUC", f"{pix_m.get('pixel_roc_auc', 0.0):.4f}")
        with p2:
            st.metric("Pixel F1 (Dice)", f"{pix_m.get('pixel_f1_dice', 0.0):.4f}")
        with p3:
            st.metric("Mean IoU", f"{pix_m.get('mean_iou', 0.0):.4f}")
        with p4:
            st.metric("Specificity", f"{img_m.get('specificity', 0.0) * 100:.1f}%")

        if sub_m:
            st.markdown("#### Defect Subcategory Detection Rates")
            sub_cols = st.columns(len(sub_m))
            for idx, (stype, sdata) in enumerate(sub_m.items()):
                with sub_cols[idx]:
                    if "accuracy" in sdata:
                        st.metric(f"{stype.title()} (Normal)", f"{sdata['accuracy'] * 100:.1f}%")
                    else:
                        st.metric(f"{stype.title()}", f"{sdata.get('detection_rate_recall', 0.0) * 100:.1f}%")

    else:
        st.info("Evaluation metrics report not yet generated. Run `python evaluate.py` to evaluate the model on the full test set.")

    st.markdown("---")
    st.markdown("### 📊 Diagnostic Curves & Visualizations")

    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        loss_plot = os.path.join(RESULTS_DIR, "loss_curve.png")
        if os.path.exists(loss_plot):
            st.image(loss_plot, caption="Training vs Validation Loss (MSE)", use_container_width=True)
        else:
            st.info("Loss curve plot will appear here after training (`python train.py`).")

    with col_plot2:
        roc_plot = os.path.join(RESULTS_DIR, "roc_curve.png")
        if os.path.exists(roc_plot):
            st.image(roc_plot, caption="Receiver Operating Characteristic (ROC) Curve", use_container_width=True)
        else:
            st.info("ROC curve plot will appear here after evaluation (`python evaluate.py`).")

    cm_plot = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    if os.path.exists(cm_plot):
        st.image(cm_plot, caption="Image Classification Confusion Matrix", width=500)

    st.markdown("---")
    st.markdown("""
    ### ⚠️ Limitations of Reconstruction-Based Anomaly Detection
    While Convolutional Autoencoders are powerful unsupervised detectors, real-world deployment must account for specific limitations:
    
    1. **Reconstruction of High-Frequency Defects:**
       If an autoencoder has excessive capacity (too many filters or too wide a bottleneck), it may partially reconstruct certain defects (especially small blurred spots), reducing the residual error and leading to false negatives.
    
    2. **False Positives on Normal Variations & Lighting Glare:**
       Reflective surfaces (such as glass bottles or polished metal) exhibit specular highlights that vary with lighting angle. The autoencoder may struggle to reconstruct bright glares, flagging harmless reflections as defects.
    
    3. **Blurry Reconstructions from Pure MSE Loss:**
       Mean Squared Error assumes Gaussian pixel noise and tends to average out fine textures, creating slightly soft edges in reconstructed outputs. Combining MSE with SSIM or perceptual loss alleviates this effect.
    
    4. **Image-Level vs. Pixel-Level Discrepancy:**
       An image may be correctly classified as defective (image-level ROC-AUC > 0.95), but fine pixel-level mask boundaries may suffer from dilation due to transposed convolution upsampling artifacts.
    """)


# ==========================================
# PAGE 4: ABOUT & ARCHITECTURE
# ==========================================
elif menu == "ℹ️ About & Architecture":
    st.markdown('<div class="main-title">About the Project & Technical Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">B.Tech Deep Learning Course Mini Project Documentation</div>', unsafe_allow_html=True)

    tab_arch, tab_dataset, tab_viva = st.tabs(["🏗️ Network Architecture", "📦 Dataset Details", "🎓 Viva & DL Concepts"])

    with tab_arch:
        st.markdown("""
        ### Deep Convolutional Autoencoder Architecture
        The network employs a symmetrical encoder-decoder topology designed for 256×256 RGB input tensors:

        ```
        INPUT: [256 x 256 x 3]
          │
          ├── Conv2D(32, 3x3, stride=2) + BatchNorm + ReLU  ──> [128 x 128 x 32]
          ├── Conv2D(64, 3x3, stride=2) + BatchNorm + ReLU  ──> [64 x 64 x 64]
          ├── Conv2D(128, 3x3, stride=2) + BatchNorm + ReLU ──> [32 x 32 x 128]
          ├── Conv2D(256, 3x3, stride=2) + BatchNorm + ReLU ──> [16 x 16 x 256]
          │
          └── [LATENT BOTTLENECK] Conv2D(64, 3x3, stride=2) ──> [8 x 8 x 64 = 4,096 dims]
          │
          ├── Conv2DTranspose(256, 3x3, stride=2) + BN + ReLU ──> [16 x 16 x 256]
          ├── Conv2DTranspose(128, 3x3, stride=2) + BN + ReLU ──> [32 x 32 x 128]
          ├── Conv2DTranspose(64, 3x3, stride=2) + BN + ReLU  ──> [64 x 64 x 64]
          ├── Conv2DTranspose(32, 3x3, stride=2) + BN + ReLU  ──> [128 x 128 x 32]
          │
        OUTPUT: Conv2DTranspose(3, 3x3, stride=2, Sigmoid)    ──> [256 x 256 x 3]
        ```

        #### Why Strided Convolutions over Max Pooling?
        Strided convolutions allow the network to learn its own adaptive spatial downsampling kernels rather than using a fixed non-differentiable maximum operation, preserving more gradient flow.
        """)

    with tab_dataset:
        st.markdown("""
        ### MVTec Anomaly Detection (MVTec AD) Benchmark
        - **Source:** MVTec Software GmbH (Comprehensive Industrial AD Benchmark)
        - **Category Selected:** `bottle` (also supports `screw`, `capsule`, etc.)
        - **Training Set:** Contains exclusively defect-free products (`train/good/`).
        - **Test Set:** Contains both normal products (`test/good/`) and multiple defect classes:
          - `broken_large`: Structural fractures and chipped bottle glass.
          - `broken_small`: Fine crack lines and hairline fissures.
          - `contamination`: External chemical residues, foreign particles, or dark blemishes.
        - **Ground Truth Masks:** Pixel-level binary masks annotating the exact defect footprint.
        """)

    with tab_viva:
        st.markdown("""
        ### Deep Learning Lab Syllabus Concepts Covered
        
        1. **What is an Autoencoder and why is it considered Unsupervised?**
           An Autoencoder uses input data $X$ as its own target $Y = X$. No manual human labeling of defects is required during training.
        
        2. **Why does the bottleneck (latent space) matter?**
           If the bottleneck were as wide as the input (256×256×3), the network could simply learn the trivial identity function ($f(x) = x$) without understanding the underlying structure. Compressing to 8×8×64 forces the network to discover compressed latent features representing the normal bottle surface.
        
        3. **How is the anomaly decision threshold calculated?**
           During validation on normal images, we compute the distribution of normal reconstruction errors (mean $\\mu$ and standard deviation $\\sigma$). The threshold is set using statistical quality control:
           $$\\tau = \\mu + 3\\sigma$$
           Under a Gaussian assumption, normal images will stay below $\\tau$ with 99.7% probability.
        
        4. **Difference between Image-Level Detection and Pixel-Level Localization:**
           - *Image-Level:* Binary classification deciding whether the entire product is acceptable or rejected (measured by Accuracy, ROC-AUC).
           - *Pixel-Level:* Semantic segmentation identifying which specific pixels are defective (measured by IoU and Dice Score).
        """)

st.markdown("---")
st.caption("B.Tech Deep Learning Mini Project | Industrial Surface Defect Detection Using CNN & Autoencoder")
