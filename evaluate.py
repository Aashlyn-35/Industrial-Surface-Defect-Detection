"""
evaluate.py - Quantitative Evaluation and Benchmark Script.

Computes:
1. Image-level metrics:
   - ROC-AUC score and ROC curve plot.
   - Classification Accuracy, Precision, Recall, F1-Score, Specificity.
   - Confusion matrix with breakdown by defect subcategory.
   - Threshold calibration comparison (Validation threshold vs Youden's optimal).

2. Pixel-level localization metrics:
   - Pixel Precision, Pixel Recall, Pixel F1-Score (Dice Score).
   - Intersection over Union (IoU / Jaccard Index).
   - Pixel-level ROC-AUC score.

3. Visualization:
   - ROC Curve (`results/roc_curve.png`)
   - Confusion Matrix (`results/confusion_matrix.png`)
   - Evaluation sample comparisons (`results/sample_eval_visualizations.png`)
   - Complete JSON metrics report (`results/evaluation_metrics.json`)
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

try:
    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from dataset import MVTecDatasetLoader, TARGET_IMAGE_SIZE
from predict import DefectDetector
import utils.metrics as metrics_utils
import utils.visualization as viz

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(
    data_root: str = "data",
    category: str = "bottle",
    model_path: str = "models/autoencoder_bottle.keras",
    config_path: str = "models/threshold_config.json",
    results_dir: str = "results",
    custom_threshold: float = None
) -> Dict[str, Any]:
    """
    Evaluates the trained autoencoder on the official test set.
    """
    os.makedirs(results_dir, exist_ok=True)

    detector = DefectDetector(model_path=model_path, config_path=config_path, allow_demo_fallback=True)
    if not detector.is_model_loaded():
        raise RuntimeError(
            f"Could not load model or baseline. Please check dataset or model files."
        )

    if detector.is_demo_mode:
        logger.info("Evaluating using Baseline Normal Surface Reconstructor (Demo Mode).")
    else:
        logger.info(f"Evaluating using Deep Learning Model from: {model_path}")

    # 2. Load Test Data
    loader = MVTecDatasetLoader(data_root=data_root, category=category, image_size=TARGET_IMAGE_SIZE)
    if not loader.is_dataset_present():
        loader.print_setup_instructions()
        raise FileNotFoundError(f"Test dataset not found for category '{category}'.")

    logger.info(f"Loading test dataset for '{category}'...")
    test_data = loader.load_test_data()

    images = test_data["images"]
    y_true_img = test_data["labels"]
    y_true_masks = test_data["masks"]
    defect_types = test_data["defect_types"]
    paths = test_data["paths"]

    if len(images) == 0:
        raise ValueError(f"No test images found in dataset for category '{category}'.")

    logger.info(f"Evaluating model on {len(images)} test images...")

    # 3. Perform Inference on All Test Images
    anomaly_scores = []
    y_pred_masks = []
    error_maps = []
    reconstructions = []

    for i in range(len(images)):
        res = detector.detect(
            images[i],
            custom_threshold=custom_threshold
        )
        anomaly_scores.append(res["anomaly_score"])
        y_pred_masks.append((res["mask"] > 0).astype(np.uint8))
        error_maps.append(res["error_map"])
        reconstructions.append(res["reconstructed_image"])

    anomaly_scores = np.array(anomaly_scores)
    y_pred_masks = np.array(y_pred_masks)
    error_maps = np.array(error_maps)

    # 4. Image-Level ROC & Threshold Determination
    youden_threshold, opt_idx, fpr, tpr = metrics_utils.find_optimal_threshold_youden(y_true_img, anomaly_scores)
    if HAS_SKLEARN:
        image_roc_auc = float(roc_auc_score(y_true_img, anomaly_scores)) if len(np.unique(y_true_img)) > 1 else 0.0
    else:
        image_roc_auc = metrics_utils._numpy_auc(fpr, tpr)

    # Use specified custom threshold, else calibrated config threshold, else Youden's
    eval_threshold = custom_threshold if custom_threshold is not None else detector.threshold

    img_metrics = metrics_utils.compute_image_classification_metrics(
        y_true=y_true_img,
        y_scores=anomaly_scores,
        threshold=eval_threshold
    )

    # 5. Pixel-Level Segmentation Metrics
    pixel_metrics = metrics_utils.compute_pixel_metrics(
        y_true_masks=y_true_masks,
        y_pred_masks=y_pred_masks
    )

    # Pixel-level ROC-AUC
    flat_true_masks = y_true_masks.flatten()
    flat_error_maps = error_maps.flatten()
    if len(np.unique(flat_true_masks)) > 1:
        sample_indices = np.random.choice(len(flat_true_masks), size=min(500000, len(flat_true_masks)), replace=False)
        sub_gt = flat_true_masks[sample_indices]
        sub_scores = flat_error_maps[sample_indices]
        if HAS_SKLEARN:
            pixel_roc_auc = float(roc_auc_score(sub_gt, sub_scores))
        else:
            p_fpr, p_tpr, _ = metrics_utils._numpy_roc_curve(sub_gt, sub_scores)
            pixel_roc_auc = metrics_utils._numpy_auc(p_fpr, p_tpr)
    else:
        pixel_roc_auc = 0.0
    pixel_metrics["pixel_roc_auc"] = pixel_roc_auc

    # 6. Defect Subcategory Breakdown
    subcategory_stats = {}
    unique_types = sorted(list(set(defect_types)))
    for d_type in unique_types:
        indices = [i for i, t in enumerate(defect_types) if t == d_type]
        sub_true = y_true_img[indices]
        sub_scores = anomaly_scores[indices]
        sub_pred = (sub_scores >= eval_threshold).astype(int)

        if d_type.lower() == "good":
            # Normal class: measure specificity / false positive rate
            correct = int(np.sum(sub_pred == 0))
            accuracy = float(correct / len(indices))
            subcategory_stats[d_type] = {
                "total": len(indices),
                "correct_normal": correct,
                "accuracy": accuracy
            }
        else:
            # Defect class: measure recall (detection rate)
            detected = int(np.sum(sub_pred == 1))
            recall = float(detected / len(indices))
            subcategory_stats[d_type] = {
                "total": len(indices),
                "detected_defects": detected,
                "detection_rate_recall": recall
            }

    # 7. Generate Evaluation Plots
    # A. ROC Curve
    roc_plot_path = os.path.join(results_dir, "roc_curve.png")
    viz.plot_roc_curve(fpr, tpr, image_roc_auc, optimal_idx=opt_idx, save_path=roc_plot_path)

    # B. Confusion Matrix
    y_pred_binary = (anomaly_scores >= eval_threshold).astype(int)
    cm = np.array([
        [img_metrics["true_negatives"], img_metrics["false_positives"]],
        [img_metrics["false_negatives"], img_metrics["true_positives"]]
    ])
    cm_plot_path = os.path.join(results_dir, "confusion_matrix.png")
    viz.plot_confusion_matrix(cm, class_names=["Normal", "Defective"], save_path=cm_plot_path)

    # C. Sample Visual Comparison Figures (one normal, one defect)
    comp_plot_path = os.path.join(results_dir, "sample_eval_visualizations.png")
    defect_indices = [i for i, lbl in enumerate(y_true_img) if lbl == 1]
    sample_idx = defect_indices[0] if defect_indices else 0
    viz.plot_inspection_result(
        original=images[sample_idx],
        reconstructed=reconstructions[sample_idx],
        error_map=error_maps[sample_idx],
        heatmap=viz.create_heatmap(error_maps[sample_idx] / (eval_threshold * 1.5)),
        mask=y_pred_masks[sample_idx],
        overlay=viz.overlay_mask_on_image(images[sample_idx], y_pred_masks[sample_idx]),
        anomaly_score=anomaly_scores[sample_idx],
        threshold=eval_threshold,
        status="Potential Defect" if y_true_img[sample_idx] == 1 else "Normal",
        ground_truth=y_true_masks[sample_idx],
        save_path=comp_plot_path
    )

    # 8. Save Metrics JSON Report
    report = {
        "dataset_category": category,
        "total_test_samples": len(images),
        "evaluation_threshold": float(eval_threshold),
        "youden_optimal_threshold": float(youden_threshold),
        "image_level_metrics": img_metrics,
        "pixel_level_metrics": pixel_metrics,
        "subcategory_breakdown": subcategory_stats
    }

    report_path = os.path.join(results_dir, "evaluation_metrics.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    logger.info(f"Saved evaluation metrics report to {report_path}")

    # 9. Print Academic Summary
    print("\n" + "=" * 65)
    print("             INDUSTRIAL DEFECT DETECTION EVALUATION REPORT")
    print("=" * 65)
    print(f"Product Category:               {category.upper()}")
    print(f"Total Test Images:              {len(images)}")
    print(f"Decision Threshold:             {eval_threshold:.5f} (Youden's: {youden_threshold:.5f})")
    print("-" * 65)
    print("IMAGE-LEVEL CLASSIFICATION PERFORMANCE:")
    print(f"  • ROC-AUC Score:              {img_metrics['roc_auc']:.4f}")
    print(f"  • Accuracy:                   {img_metrics['accuracy'] * 100:.2f}%")
    print(f"  • Precision:                  {img_metrics['precision'] * 100:.2f}%")
    print(f"  • Recall (Sensitivity):       {img_metrics['recall'] * 100:.2f}%")
    print(f"  • F1-Score:                   {img_metrics['f1_score']:.4f}")
    print(f"  • Specificity:                {img_metrics['specificity'] * 100:.2f}%")
    print("-" * 65)
    print("PIXEL-LEVEL DEFECT LOCALIZATION PERFORMANCE:")
    print(f"  • Pixel ROC-AUC:              {pixel_metrics['pixel_roc_auc']:.4f}")
    print(f"  • Pixel Precision:            {pixel_metrics['pixel_precision'] * 100:.2f}%")
    print(f"  • Pixel Recall:               {pixel_metrics['pixel_recall'] * 100:.2f}%")
    print(f"  • Pixel F1 / Dice Score:      {pixel_metrics['pixel_f1_dice']:.4f}")
    print(f"  • Mean Intersection-over-Union (IoU): {pixel_metrics['mean_iou']:.4f}")
    print("-" * 65)
    print("DEFECT SUBCATEGORY BREAKDOWN:")
    for stype, stats in subcategory_stats.items():
        if "accuracy" in stats:
            print(f"  • {stype:<16} (Normal):     Accuracy = {stats['accuracy'] * 100:.1f}% ({stats['correct_normal']}/{stats['total']})")
        else:
            print(f"  • {stype:<16} (Defective):  Detection Rate = {stats['detection_rate_recall'] * 100:.1f}% ({stats['detected_defects']}/{stats['total']})")
    print("=" * 65)
    print(f"Saved artifacts: {roc_plot_path}, {cm_plot_path}, {comp_plot_path}, {report_path}\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Autoencoder Defect Detection Model")
    parser.add_argument("--data-root", type=str, default="data", help="Root dataset directory")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec AD category")
    parser.add_argument("--model", type=str, default="models/autoencoder_bottle.keras", help="Path to .keras model")
    parser.add_argument("--config", type=str, default="models/threshold_config.json", help="Path to threshold config JSON")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save evaluation plots")
    parser.add_argument("--threshold", type=float, default=None, help="Custom threshold override")

    args = parser.parse_args()
    evaluate_model(
        data_root=args.data_root,
        category=args.category,
        model_path=args.model,
        config_path=args.config,
        results_dir=args.results_dir,
        custom_threshold=args.threshold
    )
