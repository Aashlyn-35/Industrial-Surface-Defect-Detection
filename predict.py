"""
predict.py - Inference and Defect Localization Pipeline.

Transforms raw inspection images into detailed defect diagnostics:
1. Preprocesses image to normalized (256, 256, 3) float32 tensor.
2. Reconstructs image using the trained Convolutional Autoencoder (or baseline normal reconstructor).
3. Computes pixel-wise absolute difference: E(x, y) = (1/3) * sum(|I_c - I_hat_c|).
4. Applies Gaussian smoothing to filter high-frequency sensor noise.
5. Computes a localized anomaly score (top-1% error percentile) to detect fine cracks.
6. Generates binary defect masks and colorized anomaly heatmaps.
7. Produces transparent overlays with highlighted defect boundaries.
"""

import os
import glob
import json
import logging
from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageFilter

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

import utils.visualization as viz

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/autoencoder_bottle.keras"
DEFAULT_CONFIG_PATH = "models/threshold_config.json"
DEFAULT_ANOMALY_THRESHOLD = 0.035
DEFAULT_PIXEL_SIGMA = 3.0


class BaselinePrototypeReconstructor:
    """
    Reference normal-surface reconstructor.
    Used for instant local demonstration if a trained deep learning model
    weights file (.keras) has not yet been generated or downloaded.
    Computes expected normal surface by aggregating defect-free training images.
    """

    def __init__(self, data_root: str = "data", category: str = "bottle"):
        self.category = category
        self.prototype = None
        self._build_prototype(data_root, category)

    def _build_prototype(self, data_root: str, category: str):
        train_good = Path(data_root) / category / "train" / "good"
        if train_good.exists():
            files = sorted(list(train_good.glob("*.png")) + list(train_good.glob("*.jpg")))[:20]
            imgs = []
            for f in files:
                try:
                    with Image.open(f) as im:
                        im = im.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
                        imgs.append(np.array(im, dtype=np.float32) / 255.0)
                except Exception:
                    pass
            if imgs:
                # Median normal surface filters out any minor random dust
                self.prototype = np.median(np.stack(imgs, axis=0), axis=0)

    def predict(self, tensor_batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        """
        Reconstructs normal surface. If prototype exists, blends with smooth input representation.
        """
        recon_batch = []
        for i in range(len(tensor_batch)):
            img = tensor_batch[i]
            if self.prototype is not None:
                # Weighted blend of input base structure with canonical normal surface
                # High-frequency defects will fail to match the normal prototype
                recon = 0.85 * self.prototype + 0.15 * img
            else:
                # Fallback: strong spatial smoothing (low-pass filter)
                pil_im = Image.fromarray((img * 255).astype(np.uint8))
                blurred = pil_im.filter(ImageFilter.GaussianBlur(radius=8))
                recon = np.array(blurred, dtype=np.float32) / 255.0
            recon_batch.append(recon)
        return np.stack(recon_batch, axis=0)


class DefectDetector:
    """
    Production inference engine for industrial surface defect detection.
    Loads trained Keras weights and threshold parameters.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        config_path: str = DEFAULT_CONFIG_PATH,
        model_instance: Optional[Any] = None,
        allow_demo_fallback: bool = True
    ):
        self.model_path = model_path
        self.config_path = config_path
        self.model = model_instance
        self.threshold = DEFAULT_ANOMALY_THRESHOLD
        self.category = "bottle"
        self.pixel_threshold = 0.05
        self.is_demo_mode = False
        self.model_backend_name = "None"

        # Load threshold config if present
        self._load_config()

        # Load Keras model if not provided
        if self.model is None and os.path.exists(self.model_path):
            self._load_model()

        # Fallback to demo mode if model weights are not yet present
        if self.model is None and allow_demo_fallback:
            self.model = BaselinePrototypeReconstructor(data_root="data", category=self.category)
            self.is_demo_mode = True
            self.model_backend_name = "Baseline Normal Surface Reconstructor (Demo Mode)"
            logger.info("Initialized Baseline Prototype Reconstructor for demonstration.")

    def _load_config(self):
        """Loads calculated decision threshold and calibration parameters."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    data = json.load(f)
                    self.threshold = data.get("recommended_threshold", self.threshold)
                    self.pixel_threshold = data.get("pixel_threshold", self.pixel_threshold)
                    self.category = data.get("category", self.category)
                    logger.info(f"Loaded config from {self.config_path}: threshold={self.threshold:.4f}")
            except Exception as e:
                logger.warning(f"Could not read config at {self.config_path}: {e}")

    def _load_model(self):
        """Loads saved Keras autoencoder model."""
        try:
            import tensorflow as tf
            from model import combined_mse_ssim_loss
            custom_objects = {"combined_mse_ssim_loss": combined_mse_ssim_loss}
            self.model = tf.keras.models.load_model(self.model_path, custom_objects=custom_objects)
            self.is_demo_mode = False
            self.model_backend_name = "Convolutional Autoencoder (TensorFlow/Keras)"
            logger.info(f"Loaded autoencoder model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            self.model = None

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def preprocess(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standardizes input into:
        1. normalized_tensor: float32 in [0.0, 1.0] of shape (1, 256, 256, 3)
        2. original_uint8: uint8 in [0, 255] of shape (256, 256, 3)
        """
        if isinstance(image_input, (str, Path)):
            if not os.path.exists(str(image_input)):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            pil_img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype == np.uint8:
                pil_img = Image.fromarray(image_input).convert("RGB")
            else:
                pil_img = Image.fromarray((np.clip(image_input, 0.0, 1.0) * 255).astype(np.uint8)).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        pil_img = pil_img.resize((256, 256), Image.Resampling.BILINEAR)
        original_uint8 = np.array(pil_img, dtype=np.uint8)
        normalized_float = (original_uint8.astype(np.float32) / 255.0)
        tensor_batch = np.expand_dims(normalized_float, axis=0)
        return tensor_batch, original_uint8

    def detect(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        custom_threshold: Optional[float] = None,
        custom_pixel_thresh: Optional[float] = None,
        smoothing_sigma: float = DEFAULT_PIXEL_SIGMA
    ) -> Dict[str, Any]:
        """
        Executes end-to-end defect detection on a single image.
        """
        if not self.is_model_loaded():
            raise RuntimeError(
                f"Autoencoder model is not loaded. Train a model first via 'python train.py' "
                f"or place weights at '{self.model_path}'."
            )

        tensor_batch, original_uint8 = self.preprocess(image_input)
        reconstructed_batch = self.model.predict(tensor_batch, verbose=0)
        reconstructed_float = reconstructed_batch[0]
        reconstructed_uint8 = (np.clip(reconstructed_float, 0.0, 1.0) * 255).astype(np.uint8)

        # 1. Pixel-wise Absolute Reconstruction Error
        input_float = tensor_batch[0]
        abs_diff = np.abs(input_float - reconstructed_float)
        raw_error_map = np.mean(abs_diff, axis=-1)

        # 2. Gaussian smoothing to remove high-frequency noise
        if smoothing_sigma > 0:
            if gaussian_filter is not None:
                error_map = gaussian_filter(raw_error_map, sigma=smoothing_sigma)
            else:
                # PIL Gaussian filter fallback
                err_pil = Image.fromarray((np.clip(raw_error_map, 0, 1) * 255).astype(np.uint8))
                err_blur = err_pil.filter(ImageFilter.GaussianBlur(radius=int(smoothing_sigma)))
                error_map = np.array(err_blur, dtype=np.float32) / 255.0
        else:
            error_map = raw_error_map

        # 3. Anomaly score calculation (top 1% highest error pixels)
        k_pixels = max(1, int(error_map.size * 0.01))
        top_k_errors = np.partition(error_map.flatten(), -k_pixels)[-k_pixels:]
        anomaly_score = float(np.mean(top_k_errors))
        mean_image_mse = float(np.mean(np.square(input_float - reconstructed_float)))

        # 4. Decision Threshold
        active_threshold = custom_threshold if custom_threshold is not None else self.threshold
        is_defective = bool(anomaly_score >= active_threshold)
        status = "Potential Defect" if is_defective else "Normal"

        # 5. Defect Mask Generation
        active_pixel_thresh = custom_pixel_thresh if custom_pixel_thresh is not None else (active_threshold * 0.7)
        binary_mask_bool = error_map >= active_pixel_thresh
        binary_mask_uint8 = (binary_mask_bool.astype(np.uint8)) * 255

        # 6. Heatmap and Overlay Generation
        display_max = max(float(np.max(error_map)), active_threshold * 1.5, 0.05)
        normalized_error = np.clip(error_map / display_max, 0.0, 1.0)
        heatmap_rgb = viz.create_heatmap(normalized_error, colormap_name="jet")

        overlay_rgb = viz.overlay_mask_on_image(
            original_uint8,
            binary_mask_uint8,
            color=(255, 30, 30),
            alpha=0.45,
            draw_contours=True
        )

        defect_pixel_count = int(np.sum(binary_mask_bool))
        total_pixels = binary_mask_bool.size
        defect_area_pct = float((defect_pixel_count / total_pixels) * 100.0)

        return {
            "status": status,
            "is_defective": is_defective,
            "anomaly_score": anomaly_score,
            "mean_mse": mean_image_mse,
            "threshold": active_threshold,
            "pixel_threshold": active_pixel_thresh,
            "defect_pixel_count": defect_pixel_count,
            "defect_area_pct": defect_area_pct,
            "original_image": original_uint8,
            "reconstructed_image": reconstructed_uint8,
            "error_map": error_map.astype(np.float32),
            "heatmap": heatmap_rgb,
            "mask": binary_mask_uint8,
            "overlay_image": overlay_rgb,
            "is_demo_mode": self.is_demo_mode,
            "model_backend_name": self.model_backend_name
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Defect Detection Inference on an Image")
    parser.add_argument("--image", type=str, required=True, help="Path to inspection image")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained .keras model")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to threshold config JSON")
    parser.add_argument("--threshold", type=float, default=None, help="Custom anomaly threshold override")
    parser.add_argument("--output", type=str, default="results/prediction_output.png", help="Path to save figure")

    args = parser.parse_args()

    detector = DefectDetector(model_path=args.model, config_path=args.config)
    result = detector.detect(args.image, custom_threshold=args.threshold)
    print(f"\n================ INSPECTION REPORT ================")
    print(f"Backend:       {result['model_backend_name']}")
    print(f"Status:        {result['status']}")
    print(f"Anomaly Score: {result['anomaly_score']:.5f} (Threshold: {result['threshold']:.5f})")
    print(f"Defect Area:   {result['defect_area_pct']:.2f}% ({result['defect_pixel_count']} px)")
    print(f"===================================================")

    viz.plot_inspection_result(
        original=result["original_image"],
        reconstructed=result["reconstructed_image"],
        error_map=result["error_map"],
        heatmap=result["heatmap"],
        mask=result["mask"],
        overlay=result["overlay_image"],
        anomaly_score=result["anomaly_score"],
        threshold=result["threshold"],
        status=result["status"],
        save_path=args.output
    )
    print(f"Visualization saved to: {args.output}")
