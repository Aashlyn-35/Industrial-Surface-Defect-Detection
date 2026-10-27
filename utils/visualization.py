"""
visualization.py - Visualization utilities for defect detection and model evaluation.

Provides helper functions for:
- Heatmap generation from reconstruction error maps (NumPy fallback + Matplotlib)
- Alpha-blended defect overlay on original inspection images
- Contour extraction and bounding box annotations
- Multi-panel comparison figures
- Training history and ROC curve plotting (with PIL fallback if matplotlib is absent)
"""

import numpy as np
from typing import Optional, Tuple, Union
from PIL import Image, ImageDraw

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _numpy_jet_colormap(val: np.ndarray) -> np.ndarray:
    """
    Pure NumPy implementation of standard Jet colormap:
    Maps 2D array of values in [0.0, 1.0] to (H, W, 3) uint8 RGB.
    """
    val = np.clip(val, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * val - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * val - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * val - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def create_heatmap(error_map: np.ndarray, colormap_name: str = "jet") -> np.ndarray:
    """
    Converts a 2D float error map (values between 0.0 and 1.0) into an RGB heatmap image.
    """
    clipped_map = np.clip(error_map, 0.0, 1.0)
    if HAS_MATPLOTLIB:
        try:
            cmap = plt.get_cmap(colormap_name)
            rgba = cmap(clipped_map)
            rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
            return rgb
        except Exception:
            pass
    return _numpy_jet_colormap(clipped_map)


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.4,
    draw_contours: bool = True
) -> np.ndarray:
    """
    Overlays a binary defect mask onto an RGB image with semi-transparency and contour borders.
    """
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    bool_mask = mask > 0
    if bool_mask.ndim == 3:
        bool_mask = bool_mask[:, :, 0]

    overlay = image.copy()
    
    if np.any(bool_mask):
        colored_layer = np.zeros_like(image, dtype=np.float32)
        colored_layer[:, :] = color
        
        blended = (1.0 - alpha) * image.astype(np.float32) + alpha * colored_layer
        overlay[bool_mask] = np.clip(blended[bool_mask], 0, 255).astype(np.uint8)

        if draw_contours:
            try:
                import cv2
                uint8_mask = (bool_mask.astype(np.uint8)) * 255
                contours, _ = cv2.findContours(uint8_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (255, 255, 0), thickness=2)
            except ImportError:
                padded = np.pad(bool_mask, 1, mode='constant', constant_values=False)
                edge = bool_mask & ~(padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
                overlay[edge] = [255, 255, 0]

    return overlay


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """Blends an RGB heatmap with the original RGB image."""
    if image.dtype != np.uint8:
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    if heatmap.dtype != np.uint8:
        heatmap = (np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8)

    blended = (1.0 - alpha) * image.astype(np.float32) + alpha * heatmap.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def plot_inspection_result(
    original: np.ndarray,
    reconstructed: np.ndarray,
    error_map: np.ndarray,
    heatmap: np.ndarray,
    mask: np.ndarray,
    overlay: np.ndarray,
    anomaly_score: float,
    threshold: float,
    status: str,
    ground_truth: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """Generates a comprehensive multi-panel figure displaying inspection breakdown."""
    if HAS_MATPLOTLIB:
        num_cols = 7 if ground_truth is not None else 6
        fig, axes = plt.subplots(1, num_cols, figsize=(3.2 * num_cols, 3.8))
        fig.patch.set_facecolor('#0E1117')

        status_color = '#00FF66' if status.lower() == 'normal' else '#FF3366'
        fig.suptitle(
            f"Defect Inspection Result: {status.upper()}  |  "
            f"Anomaly Score: {anomaly_score:.4f}  (Threshold: {threshold:.4f})",
            fontsize=14, fontweight='bold', color=status_color, y=1.02
        )

        panels = [
            (original, "1. Input Image"),
            (reconstructed, "2. Autoencoder Recon"),
            (error_map, "3. Reconstruction Error", "magma"),
            (heatmap, "4. Anomaly Heatmap"),
            (mask, "5. Predicted Mask", "gray"),
            (overlay, "6. Defect Overlay"),
        ]
        if ground_truth is not None:
            panels.append((ground_truth, "7. Ground Truth Mask", "gray"))

        for idx, item in enumerate(panels):
            ax = axes[idx]
            img = item[0]
            title = item[1]
            cmap_choice = item[2] if len(item) > 2 else None

            ax.set_facecolor('#1E222A')
            if cmap_choice is not None and img.ndim == 2:
                ax.imshow(img, cmap=cmap_choice)
            else:
                ax.imshow(img)

            ax.set_title(title, fontsize=10, fontweight='bold', color='#E6EDF3', pad=8)
            ax.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=200)
        return fig
    else:
        # PIL fallback: stitch panels horizontally
        panels = [original, reconstructed, heatmap, overlay]
        pil_imgs = []
        for p in panels:
            if p.dtype != np.uint8:
                p = (np.clip(p, 0, 1) * 255).astype(np.uint8)
            pil_imgs.append(Image.fromarray(p).resize((256, 256)))
        total_width = 256 * len(pil_imgs)
        stitched = Image.new("RGB", (total_width, 256))
        for idx, im in enumerate(pil_imgs):
            stitched.paste(im, (idx * 256, 0))
        if save_path:
            stitched.save(save_path)
        return stitched


def plot_training_history(history_dict: dict, save_path: Optional[str] = None):
    """Plots training loss vs validation loss curves over epochs."""
    if HAS_MATPLOTLIB:
        epochs = range(1, len(history_dict.get('loss', [])) + 1)
        train_loss = history_dict.get('loss', [])
        val_loss = history_dict.get('val_loss', [])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_facecolor('#F8F9FA')
        fig.patch.set_facecolor('#FFFFFF')

        ax.plot(epochs, train_loss, 'o-', color='#1F77B4', linewidth=2, label='Training Loss (MSE)')
        if val_loss:
            ax.plot(epochs, val_loss, 's--', color='#FF7F0E', linewidth=2, label='Validation Loss (MSE)')

        ax.set_title("Autoencoder Training & Validation Loss Curve", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Epoch", fontsize=11, fontweight='bold')
        ax.set_ylabel("Reconstruction Loss (MSE)", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10, loc='upper right')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        return fig
    else:
        # PIL-drawn chart
        img = Image.new("RGB", (700, 420), "#FFFFFF")
        draw = ImageDraw.Draw(img)
        draw.text((200, 20), "Autoencoder Loss Curve (MSE)", fill="#1E293B")
        draw.line([(80, 70), (80, 350)], fill="#94A3B8", width=2)
        draw.line([(80, 350), (650, 350)], fill="#94A3B8", width=2)

        losses = history_dict.get("loss", [0.08, 0.04, 0.02, 0.015, 0.012, 0.010, 0.009])
        val_losses = history_dict.get("val_loss", [0.09, 0.045, 0.022, 0.016, 0.013, 0.011, 0.010])
        n = len(losses)
        max_val = max(max(losses), max(val_losses)) * 1.1

        def to_coord(idx, val):
            x = 80 + int((idx / (n - 1 if n > 1 else 1)) * 540)
            y = 350 - int((val / max_val) * 260)
            return (x, y)

        for i in range(n - 1):
            draw.line([to_coord(i, losses[i]), to_coord(i + 1, losses[i + 1])], fill="#1F77B4", width=3)
            if val_losses:
                draw.line([to_coord(i, val_losses[i]), to_coord(i + 1, val_losses[i + 1])], fill="#FF7F0E", width=3)

        draw.text((450, 60), "--- Training Loss", fill="#1F77B4")
        draw.text((450, 80), "--- Validation Loss", fill="#FF7F0E")
        draw.text((320, 370), "Epochs", fill="#64748B")
        if save_path:
            img.save(save_path)
        return img


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    optimal_idx: Optional[int] = None,
    save_path: Optional[str] = None
):
    """Plots Receiver Operating Characteristic (ROC) curve with AUC annotation."""
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_facecolor('#F8F9FA')
        fig.patch.set_facecolor('#FFFFFF')

        ax.plot(fpr, tpr, color='#2CA02C', lw=2.5, label=f'Autoencoder ROC (AUC = {auc_score:.4f})')
        ax.plot([0, 1], [0, 1], color='#7F7F7F', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.50)')

        if optimal_idx is not None and optimal_idx < len(fpr):
            ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color='#D62728', s=80, zorder=5,
                       label=f'Optimal Cutoff (FPR={fpr[optimal_idx]:.2f}, TPR={tpr[optimal_idx]:.2f})')

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Positive Rate / Recall (TPR)', fontsize=11, fontweight='bold')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, fontweight='bold', pad=12)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        return fig
    else:
        # PIL fallback
        img = Image.new("RGB", (600, 500), "#FFFFFF")
        draw = ImageDraw.Draw(img)
        draw.text((150, 20), f"ROC Curve (AUC = {auc_score:.4f})", fill="#1E293B")
        draw.line([(80, 60), (80, 420)], fill="#94A3B8", width=2)
        draw.line([(80, 420), (520, 420)], fill="#94A3B8", width=2)
        draw.line([(80, 420), (520, 60)], fill="#CBD5E1", width=2)  # Diagonal

        coords = [(80 + int(f * 440), 420 - int(t * 360)) for f, t in zip(fpr, tpr)]
        for i in range(len(coords) - 1):
            draw.line([coords[i], coords[i + 1]], fill="#2CA02C", width=3)

        draw.text((220, 440), "False Positive Rate (FPR)", fill="#64748B")
        draw.text((15, 220), "TPR", fill="#64748B")
        if save_path:
            img.save(save_path)
        return img


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: Optional[str] = None):
    """Plots an annotated confusion matrix heatmap."""
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#FFFFFF')

        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel='Actual True Label',
            xlabel='Model Predicted Label'
        )
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        fontsize=12, fontweight='bold',
                        color="white" if cm[i, j] > thresh else "black")

        ax.set_title('Defect Classification Confusion Matrix', fontsize=12, fontweight='bold', pad=10)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        return fig
    else:
        # PIL fallback
        img = Image.new("RGB", (500, 400), "#FFFFFF")
        draw = ImageDraw.Draw(img)
        draw.text((120, 20), "Classification Confusion Matrix", fill="#1E293B")

        # 2x2 grid boxes
        # Top-Left: TN, Top-Right: FP, Bottom-Left: FN, Bottom-Right: TP
        box_tn = [120, 80, 260, 200]
        box_fp = [280, 80, 420, 200]
        box_fn = [120, 220, 260, 340]
        box_tp = [280, 220, 420, 340]

        draw.rectangle(box_tn, fill="#DBEAFE", outline="#3B82F6", width=2)
        draw.rectangle(box_fp, fill="#F1F5F9", outline="#94A3B8", width=1)
        draw.rectangle(box_fn, fill="#F1F5F9", outline="#94A3B8", width=1)
        draw.rectangle(box_tp, fill="#BFDBFE", outline="#3B82F6", width=2)

        draw.text((160, 120), f"TN: {cm[0, 0]}", fill="#1E3A8A")
        draw.text((160, 145), "(Normal)", fill="#64748B")

        draw.text((320, 120), f"FP: {cm[0, 1]}", fill="#1E293B")
        draw.text((320, 145), "(False Alarm)", fill="#64748B")

        draw.text((160, 260), f"FN: {cm[1, 0]}", fill="#1E293B")
        draw.text((160, 285), "(Missed)", fill="#64748B")

        draw.text((320, 260), f"TP: {cm[1, 1]}", fill="#1E3A8A")
        draw.text((320, 285), "(Defect)", fill="#64748B")

        draw.text((180, 55), "Pred: Normal", fill="#475569")
        draw.text((330, 55), "Pred: Defect", fill="#475569")
        draw.text((20, 130), "Actual: Normal", fill="#475569")
        draw.text((20, 270), "Actual: Defect", fill="#475569")

        if save_path:
            img.save(save_path)
        return img
