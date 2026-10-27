"""
dataset.py - Data loader and preprocessor for the MVTec Anomaly Detection (MVTec AD) dataset.

Features:
- Loads training images (strictly 'good' normal images).
- Splits training set into train and validation sets with reproducible seeding.
- Loads test images across all subcategories ('good' and all defect types).
- Loads corresponding ground-truth defect masks (returns zero-mask for 'good' test images).
- Resizes images to (256, 256) and normalizes pixel values to [0.0, 1.0].
- Validates file integrity and handles corrupted or missing files safely.
- Provides a synthetic sample generator for immediate testing without downloading the full 5GB archive.
"""

import os
import glob
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Official MVTec AD download link and reference
MVTEC_AD_URL = "https://www.mvtec.com/company/research/datasets/mvtec-ad"
DEFAULT_CATEGORY = "bottle"
TARGET_IMAGE_SIZE = (256, 256)


def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int] = TARGET_IMAGE_SIZE) -> np.ndarray:
    """
    Loads an image file, converts to RGB, resizes to target_size,
    and normalizes pixel values from [0, 255] to [0.0, 1.0] float32.

    Args:
        image_path: Path to image file (.png, .jpg, etc.)
        target_size: (width, height) tuple

    Returns:
        numpy.ndarray of shape (height, width, 3), dtype=np.float32, range [0.0, 1.0]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize(target_size, Image.Resampling.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            return arr
    except Exception as e:
        raise ValueError(f"Failed to read image '{image_path}': {e}")


def load_and_preprocess_mask(mask_path: Optional[str], target_size: Tuple[int, int] = TARGET_IMAGE_SIZE) -> np.ndarray:
    """
    Loads a binary ground truth mask image, resizes it, and binarizes it.
    If mask_path is None or file does not exist, returns an all-zero mask.

    Returns:
        numpy.ndarray of shape (height, width), dtype=np.uint8 with values {0, 1}.
    """
    if mask_path is None or not os.path.exists(mask_path):
        return np.zeros(target_size, dtype=np.uint8)

    try:
        with Image.open(mask_path) as img:
            img = img.convert("L")  # Grayscale
            img = img.resize(target_size, Image.Resampling.NEAREST)
            arr = np.array(img, dtype=np.uint8)
            return (arr > 127).astype(np.uint8)
    except Exception as e:
        logger.warning(f"Failed to read mask '{mask_path}', returning blank mask: {e}")
        return np.zeros(target_size, dtype=np.uint8)


class MVTecDatasetLoader:
    """
    Official MVTec AD Dataset Loader for a chosen category (e.g. 'bottle').

    Expected directory structure:
    <data_root>/<category>/
        ├── train/
        │   └── good/
        │       ├── 000.png ...
        ├── test/
        │   ├── good/
        │   ├── broken_large/
        │   ├── broken_small/
        │   └── contamination/
        └── ground_truth/
            ├── broken_large/
            │   ├── 000_mask.png ...
            ├── broken_small/
            └── contamination/
    """

    def __init__(self, data_root: str = "data", category: str = DEFAULT_CATEGORY, image_size: Tuple[int, int] = TARGET_IMAGE_SIZE):
        self.data_root = Path(data_root)
        self.category = category
        self.category_path = self.data_root / category
        self.image_size = image_size

    def is_dataset_present(self) -> bool:
        """Checks if train/good folder exists and contains images."""
        train_good = self.category_path / "train" / "good"
        if not train_good.exists():
            return False
        images = list(train_good.glob("*.png")) + list(train_good.glob("*.jpg"))
        return len(images) > 0

    def print_setup_instructions(self):
        """Displays clear instructions for downloading and extracting MVTec AD."""
        msg = f"""
========================================================================
[MVTec AD Dataset Missing]
Category '{self.category}' not found at: {self.category_path.resolve()}

To download the official dataset:
1. Visit: {MVTEC_AD_URL}
2. Download '{self.category}.tar.xz' (approx 100-300 MB per category)
3. Extract it into your project under: {self.data_root.resolve()}/{self.category}/
   So the path structure looks like:
     {self.data_root.resolve()}/{self.category}/train/good/
     {self.data_root.resolve()}/{self.category}/test/
     {self.data_root.resolve()}/{self.category}/ground_truth/

Alternatively, run:
    python dataset.py --generate-samples
to generate high-quality synthetic sample images for rapid development and testing!
========================================================================
"""
        print(msg)

    def load_train_data(self, val_split: float = 0.15, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Loads normal training images from `<category>/train/good`.
        Splits into train and validation arrays.

        Returns:
            (train_images, val_images, train_paths, val_paths)
            Where images are arrays of shape (N, H, W, 3) in [0.0, 1.0].
        """
        if not self.is_dataset_present():
            self.print_setup_instructions()
            raise FileNotFoundError(f"Training dataset directory not found: {self.category_path / 'train' / 'good'}")

        train_good_dir = self.category_path / "train" / "good"
        image_files = sorted(glob.glob(str(train_good_dir / "*.png")) + glob.glob(str(train_good_dir / "*.jpg")))

        if not image_files:
            raise ValueError(f"No valid image files found in {train_good_dir}")

        logger.info(f"Loading {len(image_files)} normal training images from {train_good_dir}")

        images = []
        valid_paths = []
        for p in image_files:
            try:
                arr = load_and_preprocess_image(p, self.image_size)
                images.append(arr)
                valid_paths.append(p)
            except Exception as e:
                logger.warning(f"Skipping corrupt training image {p}: {e}")

        images_np = np.stack(images, axis=0)

        # Reproducible validation split
        np.random.seed(random_seed)
        n_samples = len(images_np)
        indices = np.random.permutation(n_samples)

        n_val = int(n_samples * val_split)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        train_images = images_np[train_idx]
        val_images = images_np[val_idx]
        train_paths = [valid_paths[i] for i in train_idx]
        val_paths = [valid_paths[i] for i in val_idx]

        logger.info(f"Dataset split: {len(train_images)} training images, {len(val_images)} validation images.")
        return train_images, val_images, train_paths, val_paths

    def load_test_data(self) -> Dict[str, any]:
        """
        Loads all test images and corresponding ground-truth defect masks.

        Returns a dictionary:
            {
                'images': (N, H, W, 3) float32,
                'labels': (N,) int (0 = normal, 1 = defective),
                'masks': (N, H, W) uint8 {0, 1},
                'defect_types': (N,) list of strings (e.g. 'good', 'broken_large'),
                'paths': (N,) list of image file paths
            }
        """
        test_dir = self.category_path / "test"
        gt_dir = self.category_path / "ground_truth"

        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        defect_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
        if not defect_dirs:
            raise ValueError(f"No defect subdirectories found in {test_dir}")

        images_list = []
        labels_list = []
        masks_list = []
        types_list = []
        paths_list = []

        for defect_type_path in defect_dirs:
            defect_type = defect_type_path.name
            is_anomaly = (defect_type.lower() != "good")
            label = 1 if is_anomaly else 0

            image_files = sorted(list(defect_type_path.glob("*.png")) + list(defect_type_path.glob("*.jpg")))
            logger.info(f"Found {len(image_files)} test samples for type '{defect_type}' (anomaly={is_anomaly})")

            for img_p in image_files:
                try:
                    img_arr = load_and_preprocess_image(str(img_p), self.image_size)

                    # Look for matching ground truth mask
                    mask_arr = None
                    if is_anomaly and gt_dir.exists():
                        mask_name_png = f"{img_p.stem}_mask.png"
                        mask_p = gt_dir / defect_type / mask_name_png
                        if not mask_p.exists():
                            # Alternate naming without _mask
                            mask_p = gt_dir / defect_type / f"{img_p.stem}.png"
                        if mask_p.exists():
                            mask_arr = load_and_preprocess_mask(str(mask_p), self.image_size)

                    if mask_arr is None:
                        mask_arr = np.zeros(self.image_size, dtype=np.uint8)

                    images_list.append(img_arr)
                    labels_list.append(label)
                    masks_list.append(mask_arr)
                    types_list.append(defect_type)
                    paths_list.append(str(img_p))

                except Exception as e:
                    logger.warning(f"Error loading test image {img_p}: {e}")

        logger.info(f"Loaded total {len(images_list)} test samples across {len(defect_dirs)} categories.")
        return {
            "images": np.stack(images_list, axis=0) if images_list else np.empty((0, *self.image_size, 3), dtype=np.float32),
            "labels": np.array(labels_list, dtype=int),
            "masks": np.stack(masks_list, axis=0) if masks_list else np.empty((0, *self.image_size), dtype=np.uint8),
            "defect_types": types_list,
            "paths": paths_list
        }


def generate_synthetic_mvtec_dataset(
    target_dir: str = "data",
    category: str = "bottle",
    num_train: int = 40,
    num_test_per_type: int = 10
):
    """
    Generates a realistic synthetic mini-dataset mimicking MVTec AD 'bottle'
    with normal glass bottles and several defect classes (broken_large, broken_small, contamination).
    This enables immediate local testing, debugging, and demonstration without needing large downloads!
    """
    from PIL import ImageDraw

    base_path = Path(target_dir) / category
    train_good = base_path / "train" / "good"
    test_good = base_path / "test" / "good"
    test_broken_l = base_path / "test" / "broken_large"
    test_broken_s = base_path / "test" / "broken_small"
    test_contam = base_path / "test" / "contamination"

    gt_broken_l = base_path / "ground_truth" / "broken_large"
    gt_broken_s = base_path / "ground_truth" / "broken_small"
    gt_contam = base_path / "ground_truth" / "contamination"

    for p in [train_good, test_good, test_broken_l, test_broken_s, test_contam,
              gt_broken_l, gt_broken_s, gt_contam]:
        p.mkdir(parents=True, exist_ok=True)

    print(f"Creating synthetic MVTec AD '{category}' dataset in: {base_path.resolve()} ...")

    def draw_base_bottle(width=256, height=256, seed=None):
        """Draws a clean glass bottle on an industrial dark inspection background."""
        rng = np.random.RandomState(seed)
        # Background: dark metallic inspection stage with subtle texture
        bg_val = rng.randint(20, 30)
        img = Image.new("RGB", (width, height), (bg_val, bg_val, bg_val + 2))
        draw = ImageDraw.Draw(img)

        # Bottle neck and body coordinates
        # Center x = 128, body width = 80, body height = 140
        body_box = [88, 90, 168, 230]
        neck_box = [114, 30, 142, 90]
        cap_box = [110, 20, 146, 35]

        # Draw glass body gradient/fill
        glass_color = (rng.randint(60, 75), rng.randint(140, 160), rng.randint(180, 205))
        draw.rectangle(body_box, fill=glass_color, outline=(180, 220, 240), width=2)
        draw.rectangle(neck_box, fill=glass_color, outline=(180, 220, 240), width=2)
        draw.rectangle(cap_box, fill=(180, 40, 40), outline=(220, 220, 220), width=1)  # Red cap

        # Glass reflections (vertical highlight lines)
        draw.line([100, 95, 100, 225], fill=(230, 245, 255), width=3)
        draw.line([106, 95, 106, 225], fill=(200, 230, 250), width=1)
        return img

    # 1. Generate Training Normal Images
    for i in range(num_train):
        img = draw_base_bottle(seed=1000 + i)
        img.save(train_good / f"{i:03d}.png")

    # 2. Generate Test Normal Images
    for i in range(num_test_per_type):
        img = draw_base_bottle(seed=2000 + i)
        img.save(test_good / f"{i:03d}.png")

    # 3. Generate Test Defect: Contamination (dark spot on glass)
    for i in range(num_test_per_type):
        img = draw_base_bottle(seed=3000 + i)
        mask = Image.new("L", (256, 256), 0)
        draw_img = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)

        cx, cy = 120 + (i % 3) * 10, 130 + (i % 4) * 15
        r = 8 + (i % 4)
        draw_img.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(15, 15, 15), outline=(0, 0, 0))
        draw_mask.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

        img.save(test_contam / f"{i:03d}.png")
        mask.save(gt_contam / f"{i:03d}_mask.png")

    # 4. Generate Test Defect: Broken Large (chipped bottle body/neck)
    for i in range(num_test_per_type):
        img = draw_base_bottle(seed=4000 + i)
        mask = Image.new("L", (256, 256), 0)
        draw_img = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)

        # Chipped jagged polygon on side of bottle
        poly = [(160, 110 + i * 4), (175, 125 + i * 4), (168, 145 + i * 4), (150, 130 + i * 4)]
        draw_img.polygon(poly, fill=(25, 25, 25), outline=(240, 240, 240))
        draw_mask.polygon(poly, fill=255)

        img.save(test_broken_l / f"{i:03d}.png")
        mask.save(gt_broken_l / f"{i:03d}_mask.png")

    # 5. Generate Test Defect: Broken Small (fine crack line)
    for i in range(num_test_per_type):
        img = draw_base_bottle(seed=5000 + i)
        mask = Image.new("L", (256, 256), 0)
        draw_img = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)

        # Crack line
        x0, y0 = 110 + (i % 4) * 5, 100 + (i % 3) * 20
        crack_points = [(x0, y0), (x0 + 8, y0 + 12), (x0 + 4, y0 + 25), (x0 + 15, y0 + 38)]
        draw_img.line(crack_points, fill=(255, 255, 255), width=2)
        draw_mask.line(crack_points, fill=255, width=4)

        img.save(test_broken_s / f"{i:03d}.png")
        mask.save(gt_broken_s / f"{i:03d}_mask.png")

    print(f"Synthetic dataset generation complete! Created {num_train} train images and {num_test_per_type * 4} test samples.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MVTec AD Dataset Loader & Utilities")
    parser.add_argument("--data-root", type=str, default="data", help="Root directory for dataset")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec AD category (e.g. bottle, screw)")
    parser.add_argument("--generate-samples", action="store_true", help="Generate synthetic test images for demonstration")
    parser.add_argument("--check", action="store_true", help="Check if dataset exists and display status")

    args = parser.parse_args()
    loader = MVTecDatasetLoader(data_root=args.data_root, category=args.category)

    if args.generate_samples:
        generate_synthetic_mvtec_dataset(target_dir=args.data_root, category=args.category)
    elif args.check or True:
        if loader.is_dataset_present():
            print(f"Dataset for category '{args.category}' is present and verified.")
        else:
            loader.print_setup_instructions()
