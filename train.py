"""
train.py - Training pipeline for Industrial Surface Defect Autoencoder.

Key Workflow:
1. Loads defect-free ('good') training images from MVTec AD dataset.
2. Splits dataset into training and validation subsets (e.g., 85% train, 15% val).
3. Constructs Convolutional Autoencoder architecture.
4. Uses Keras callbacks:
   - EarlyStopping: Halts training if validation loss plateaus to prevent overfitting.
   - ModelCheckpoint: Saves the best model checkpoint weights automatically.
   - ReduceLROnPlateau: Decays learning rate dynamically on plateau.
5. Saves the final trained model (.keras).
6. Saves training history (loss, val_loss) to JSON.
7. Generates and saves training vs. validation loss curve plot.
8. Calibrates anomaly decision threshold based on validation error distribution (mean + 3*std).
"""

import os
import json
import argparse
import logging
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from dataset import MVTecDatasetLoader, TARGET_IMAGE_SIZE
from model import build_autoencoder
import utils.visualization as viz

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_autoencoder(
    data_root: str = "data",
    category: str = "bottle",
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    val_split: float = 0.15,
    loss_type: str = "mse",
    output_dir: str = "models",
    results_dir: str = "results"
):
    """
    Executes the training and validation pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Dataset Loading
    loader = MVTecDatasetLoader(data_root=data_root, category=category, image_size=TARGET_IMAGE_SIZE)
    if not loader.is_dataset_present():
        loader.print_setup_instructions()
        raise FileNotFoundError(
            f"Dataset for category '{category}' not found at '{loader.category_path}'. "
            "Please download the dataset or run 'python dataset.py --generate-samples' to generate synthetic samples."
        )

    logger.info(f"Loading training data for category '{category}'...")
    train_images, val_images, train_paths, val_paths = loader.load_train_data(val_split=val_split)
    logger.info(f"Loaded {len(train_images)} training images and {len(val_images)} validation images.")

    # 2. Build Autoencoder
    logger.info("Instantiating Convolutional Autoencoder...")
    autoencoder = build_autoencoder(
        input_shape=(TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 3),
        learning_rate=learning_rate,
        loss_type=loss_type
    )
    autoencoder.summary(print_fn=logger.info)

    # 3. Setup Callbacks
    model_save_path = os.path.join(output_dir, f"autoencoder_{category}.keras")
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1,
            restore_best_weights=True,
            mode="min"
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
            mode="min"
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
            mode="min"
        )
    ]

    # 4. Train Autoencoder (Input = Target because Autoencoder reconstructs input)
    logger.info(f"Starting Autoencoder training for {epochs} epochs (Batch size: {batch_size})...")
    history = autoencoder.fit(
        x=train_images,
        y=train_images,
        validation_data=(val_images, val_images),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True
    )

    # 5. Save History & Plot Loss Curves
    history_dict = {
        "loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
        "epochs": len(history.history["loss"]),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "loss_type": loss_type,
        "category": category
    }

    history_json_path = os.path.join(results_dir, "training_history.json")
    with open(history_json_path, "w") as f:
        json.dump(history_dict, f, indent=4)
    logger.info(f"Saved training history to {history_json_path}")

    loss_plot_path = os.path.join(results_dir, "loss_curve.png")
    viz.plot_training_history(history_dict, save_path=loss_plot_path)
    logger.info(f"Saved loss curves plot to {loss_plot_path}")

    # 6. Calibrate Anomaly Threshold on Normal Validation Images
    # The threshold is computed using statistical outlier analysis (mu + 3 * sigma)
    # on normal images so normal surfaces produce < 1% false positives.
    logger.info("Calibrating anomaly threshold using validation set...")
    val_preds = autoencoder.predict(val_images, batch_size=batch_size, verbose=0)
    
    val_scores = []
    val_pixel_errors = []
    for i in range(len(val_images)):
        abs_err = np.mean(np.abs(val_images[i] - val_preds[i]), axis=-1)
        val_pixel_errors.append(abs_err)
        # Top 1% pixel mean
        k_pix = max(1, int(abs_err.size * 0.01))
        top_k = np.partition(abs_err.flatten(), -k_pix)[-k_pix:]
        val_scores.append(float(np.mean(top_k)))

    val_scores = np.array(val_scores)
    mean_val_score = float(np.mean(val_scores))
    std_val_score = float(np.std(val_scores))
    
    # 3-sigma statistical control limit
    recommended_threshold = float(mean_val_score + 3.0 * std_val_score)
    # 98th percentile of individual pixel errors across validation set
    all_pixel_errs = np.concatenate([p.flatten() for p in val_pixel_errors])
    recommended_pixel_thresh = float(np.percentile(all_pixel_errs, 99.0))

    threshold_config = {
        "category": category,
        "mean_normal_score": mean_val_score,
        "std_normal_score": std_val_score,
        "recommended_threshold": recommended_threshold,
        "pixel_threshold": recommended_pixel_thresh,
        "validation_samples_count": len(val_images),
        "model_file": f"autoencoder_{category}.keras"
    }

    config_save_path = os.path.join(output_dir, "threshold_config.json")
    with open(config_save_path, "w") as f:
        json.dump(threshold_config, f, indent=4)
    logger.info(f"Threshold calibrated: mu={mean_val_score:.5f}, sigma={std_val_score:.5f} => Threshold={recommended_threshold:.5f}")
    logger.info(f"Threshold config saved to {config_save_path}")

    print("\n" + "=" * 60)
    print("           TRAINING & CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"Model File:          {model_save_path}")
    print(f"Loss Curve Plot:     {loss_plot_path}")
    print(f"Calibrated Threshold: {recommended_threshold:.5f}")
    print(f"Pixel Threshold:     {recommended_pixel_thresh:.5f}")
    print("=" * 60 + "\n")

    return autoencoder, history_dict, threshold_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Autoencoder for Surface Defect Detection")
    parser.add_argument("--data-root", type=str, default="data", help="Root dataset directory")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec AD category (e.g. bottle, screw)")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split fraction")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "combined"], help="Loss function")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for saved models")
    parser.add_argument("--results-dir", type=str, default="results", help="Output directory for plots & logs")

    args = parser.parse_args()
    train_autoencoder(
        data_root=args.data_root,
        category=args.category,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        loss_type=args.loss,
        output_dir=args.output_dir,
        results_dir=args.results_dir
    )
