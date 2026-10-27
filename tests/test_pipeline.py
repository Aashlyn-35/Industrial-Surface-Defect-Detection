"""
test_pipeline.py - Comprehensive Unit Tests for Data Preprocessing,
Metrics, Model Shapes, and Defect Detection Pipeline.
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image
import dataset
import utils.metrics as metrics_utils
import utils.visualization as viz


class TestDefectDetectionPipeline(unittest.TestCase):
    """Unit tests for the end-to-end defect detection components."""

    def setUp(self):
        self.test_dir = "tests/test_artifacts"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a sample test image
        self.sample_img_path = os.path.join(self.test_dir, "sample.png")
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(arr).save(self.sample_img_path)

        # Create a sample mask image
        self.sample_mask_path = os.path.join(self.test_dir, "sample_mask.png")
        mask_arr = np.zeros((100, 100), dtype=np.uint8)
        mask_arr[20:40, 20:40] = 255
        Image.fromarray(mask_arr).save(self.sample_mask_path)

    def tearDown(self):
        # Clean up temporary test files
        if os.path.exists(self.sample_img_path):
            os.remove(self.sample_img_path)
        if os.path.exists(self.sample_mask_path):
            os.remove(self.sample_mask_path)
        if os.path.exists(self.test_dir):
            try:
                os.rmdir(self.test_dir)
            except OSError:
                pass

    def test_image_preprocessing(self):
        """Checks image normalization range [0.0, 1.0] and target resize shape (256, 256, 3)."""
        img_arr = dataset.load_and_preprocess_image(self.sample_img_path, target_size=(256, 256))
        self.assertEqual(img_arr.shape, (256, 256, 3))
        self.assertEqual(img_arr.dtype, np.float32)
        self.assertGreaterEqual(float(np.min(img_arr)), 0.0)
        self.assertLessEqual(float(np.max(img_arr)), 1.0)

    def test_mask_preprocessing(self):
        """Checks mask binarization and output shape."""
        mask = dataset.load_and_preprocess_mask(self.sample_mask_path, target_size=(256, 256))
        self.assertEqual(mask.shape, (256, 256))
        self.assertTrue(set(np.unique(mask)).issubset({0, 1}))

    def test_missing_file_safety(self):
        """Verifies safe fallback when an image or mask path is missing."""
        missing_mask = dataset.load_and_preprocess_mask("non_existent_file.png", target_size=(256, 256))
        self.assertEqual(missing_mask.shape, (256, 256))
        self.assertEqual(int(np.sum(missing_mask)), 0)

        with self.assertRaises(FileNotFoundError):
            dataset.load_and_preprocess_image("non_existent_file.png")

    def test_heatmap_generation(self):
        """Verifies heatmap output has correct dimensions and uint8 RGB channels."""
        error_map = np.random.uniform(0.0, 1.0, (256, 256)).astype(np.float32)
        heatmap = viz.create_heatmap(error_map, colormap_name="jet")
        self.assertEqual(heatmap.shape, (256, 256, 3))
        self.assertEqual(heatmap.dtype, np.uint8)

    def test_overlay_creation(self):
        """Checks overlay blending between image and mask."""
        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:100, 50:100] = 255

        overlay = viz.overlay_mask_on_image(img, mask, color=(255, 0, 0), alpha=0.5)
        self.assertEqual(overlay.shape, (256, 256, 3))
        self.assertEqual(overlay.dtype, np.uint8)
        # Verify that defect region was modified
        self.assertNotEqual(overlay[75, 75, 0], 128)

    def test_classification_metrics(self):
        """Tests image-level metric calculations."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.01, 0.02, 0.08, 0.09])
        threshold = 0.05

        metrics = metrics_utils.compute_image_classification_metrics(y_true, y_scores, threshold)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_pixel_localization_metrics(self):
        """Tests pixel-level IoU and Dice metric calculation."""
        gt_mask = np.zeros((10, 10), dtype=np.uint8)
        gt_mask[2:6, 2:6] = 1  # 4x4 = 16 pixels

        pred_mask = np.zeros((10, 10), dtype=np.uint8)
        pred_mask[2:6, 2:6] = 1  # Perfect match

        metrics = metrics_utils.compute_pixel_metrics([gt_mask], [pred_mask])
        self.assertAlmostEqual(metrics["mean_iou"], 1.0, places=4)
        self.assertAlmostEqual(metrics["pixel_f1_dice"], 1.0, places=4)

    def test_synthetic_data_generator(self):
        """Tests synthetic MVTec AD dataset creation."""
        dataset.generate_synthetic_mvtec_dataset(
            target_dir=self.test_dir,
            category="test_bottle",
            num_train=3,
            num_test_per_type=2
        )
        loader = dataset.MVTecDatasetLoader(data_root=self.test_dir, category="test_bottle")
        self.assertTrue(loader.is_dataset_present())
        train_img, val_img, _, _ = loader.load_train_data(val_split=0.33)
        self.assertEqual(train_img.shape[1:], (256, 256, 3))
        self.assertGreater(len(train_img), 0)


if __name__ == "__main__":
    unittest.main()
