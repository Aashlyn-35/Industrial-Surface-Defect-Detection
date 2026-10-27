"""
metrics.py - Evaluation metrics for anomaly detection and defect segmentation.

Computes:
1. Image-level metrics:
   - ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
   - Confusion Matrix (TP, FP, TN, FN)
   - Precision, Recall, F1-Score, Accuracy, Specificity
   - Optimal threshold finding via Youden's J-statistic (TPR - FPR)

2. Pixel-level metrics (where ground truth masks are available):
   - Pixel Precision, Recall, F1-Score (Dice Coefficient)
   - Intersection-over-Union (IoU / Jaccard Index)
   - Pixel-level ROC-AUC
"""

import numpy as np
from typing import Dict, Tuple, Optional, List

# Graceful optional import for scikit-learn
try:
    from sklearn.metrics import (
        roc_auc_score,
        roc_curve,
        precision_recall_fscore_support,
        accuracy_score,
        confusion_matrix
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def _numpy_roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure NumPy implementation of ROC curve calculation."""
    desc_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_indices]
    y_scores_sorted = y_scores[desc_indices]

    distinct_indices = np.where(np.diff(y_scores_sorted))[0]
    threshold_indices = np.r_[distinct_indices, y_true.size - 1]

    tps = np.cumsum(y_true_sorted == 1)[threshold_indices]
    fps = np.cumsum(y_true_sorted == 0)[threshold_indices]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    tpr = tps / n_pos if n_pos > 0 else np.zeros_like(tps, dtype=float)
    fpr = fps / n_neg if n_neg > 0 else np.zeros_like(fps, dtype=float)

    thresholds = y_scores_sorted[threshold_indices]

    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[thresholds[0] + 1e-4, thresholds]
    return fpr, tpr, thresholds


def _numpy_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Computes AUC using trapezoidal numerical integration."""
    # Ensure trapezoid integration works across numpy versions
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(tpr, fpr))
    return float(np.trapz(tpr, fpr))


def compute_image_classification_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Computes comprehensive image-level classification metrics given true binary labels,
    predicted continuous anomaly scores, and a decision threshold.

    Labels convention:
      0 = Normal
      1 = Defective (Anomaly)
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)
    y_pred = (y_scores >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ROC AUC calculation
    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        if HAS_SKLEARN:
            auc_score = float(roc_auc_score(y_true, y_scores))
        else:
            fpr_arr, tpr_arr, _ = _numpy_roc_curve(y_true, y_scores)
            auc_score = _numpy_auc(fpr_arr, tpr_arr)
    else:
        auc_score = 0.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "specificity": float(specificity),
        "roc_auc": float(auc_score),
        "threshold": float(threshold),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "total_samples": int(len(y_true))
    }


def find_optimal_threshold_youden(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, int, np.ndarray, np.ndarray]:
    """
    Finds the optimal decision threshold that maximizes Youden's J statistic:
    J = TPR - FPR = Sensitivity + Specificity - 1.

    Returns:
        (optimal_threshold, best_index, fpr, tpr)
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_scores) + 2 * np.std(y_scores)), 0, np.array([0, 1]), np.array([0, 1])

    if HAS_SKLEARN:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    else:
        fpr, tpr, thresholds = _numpy_roc_curve(y_true, y_scores)

    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    optimal_thresh = float(thresholds[best_idx])
    return optimal_thresh, best_idx, fpr, tpr


def compute_pixel_metrics(
    y_true_masks: List[np.ndarray],
    y_pred_masks: List[np.ndarray]
) -> Dict[str, float]:
    """
    Computes pixel-level localization metrics:
    - Pixel Precision
    - Pixel Recall
    - Pixel F1-Score (Dice Coefficient)
    - Mean IoU (Intersection over Union)

    Args:
        y_true_masks: List or array of ground truth binary masks (0 = background, 1 = defect).
        y_pred_masks: List or array of predicted binary masks (0 = background, 1 = defect).
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    iou_scores = []

    for gt, pred in zip(y_true_masks, y_pred_masks):
        gt_bin = (gt > 0).astype(bool)
        pred_bin = (pred > 0).astype(bool)

        tp = np.logical_and(gt_bin, pred_bin).sum()
        fp = np.logical_and(~gt_bin, pred_bin).sum()
        fn = np.logical_and(gt_bin, ~pred_bin).sum()
        tn = np.logical_and(~gt_bin, ~pred_bin).sum()

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        intersection = tp
        union = tp + fp + fn
        if union == 0:
            iou_scores.append(1.0)
        else:
            iou_scores.append(float(intersection) / float(union))

    pixel_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    pixel_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    pixel_f1 = (2 * pixel_precision * pixel_recall / (pixel_precision + pixel_recall)
                if (pixel_precision + pixel_recall) > 0 else 0.0)
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

    return {
        "pixel_precision": float(pixel_precision),
        "pixel_recall": float(pixel_recall),
        "pixel_f1_dice": float(pixel_f1),
        "mean_iou": float(mean_iou),
        "total_pixels_evaluated": int(total_tp + total_fp + total_fn + total_tn)
    }
