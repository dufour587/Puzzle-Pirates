# variants.py - The Master-Class Data Augmentation Engine
#
# This version has been re-engineered for:
# - A compositional, pipeline-based approach for chaining augmentations.
# - A suite of new, highly realistic transformations.
# - A cleaner, object-oriented design for improved maintainability.

import cv2
import numpy as np
import random
from typing import Tuple, Optional, List, Dict
import logging

# Set up logging for professional-level diagnostics
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageTransform:
    """Base class for all image transformations."""
    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class RandomFlip(ImageTransform):
    """Randomly flips the image horizontally."""
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.choice([True, False]):
            return cv2.flip(img, 1)
        return img

class RandomColorJitter(ImageTransform):
    """
    Applies random brightness, contrast, and saturation changes simultaneously.
    This is a more sophisticated and realistic alternative to separate adjustments.
    """
    def __call__(self, img: np.ndarray) -> np.ndarray:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)

        # Apply random brightness and contrast
        brightness_factor = random.uniform(0.7, 1.3)
        contrast_factor = random.uniform(0.7, 1.3)
        v = np.clip(128 + contrast_factor * (v - 128), 0, 255).astype(np.uint8)
        v = np.clip(v * brightness_factor, 0, 255).astype(np.uint8)

        # Apply random saturation
        saturation_factor = random.uniform(0.7, 1.3)
        s = np.clip(s * saturation_factor, 0, 255).astype(np.uint8)

        final_hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

class AddGaussianNoise(ImageTransform):
    """Adds Gaussian noise to the image."""
    def __init__(self, severity: float = 0.05):
        self.severity = severity

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img_norm = img.astype(np.float32) / 255.0
        noise = np.random.normal(0, self.severity, img.shape).astype(np.float32)
        noisy_img = np.clip(img_norm + noise, 0, 1)
        return (noisy_img * 255.0).astype(np.uint8)

class ApplyBlur(ImageTransform):
    """Applies a Gaussian blur to the image."""
    def __init__(self, kernel_size: int = 3):
        self.kernel_size = (kernel_size, kernel_size)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(img, self.kernel_size, 0)

class RandomOcclusion(ImageTransform):
    """
    Simulates a small occlusion (e.g., a mouse cursor or UI element)
    by placing a random black box on the image.
    """
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() < 0.2:
            img_copy = img.copy()
            h, w, _ = img_copy.shape
            occ_w = random.randint(10, w // 4)
            occ_h = random.randint(10, h // 4)
            x = random.randint(0, w - occ_w)
            y = random.randint(0, h - occ_h)
            cv2.rectangle(img_copy, (x, y), (x + occ_w, y + occ_h), (0, 0, 0), -1)
            return img_copy
        return img

class Pipeline:
    """A compositional pipeline for applying a sequence of transformations."""
    def __init__(self, transforms: List[ImageTransform]):
        self.transforms = transforms

    def __call__(self, img: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            img = transform(img)
        return img

class UltimateVariantGenerator:
    """
    A powerful class for generating diverse visual variants of game assets
    to improve computer vision model robustness.
    """
    def __init__(self):
        # We now define the pipeline in the __init__ method for reusability
        self.default_pipeline = Pipeline([
            RandomFlip(),
            RandomColorJitter(),
            AddGaussianNoise(severity=random.uniform(0.01, 0.05)),
            ApplyBlur(kernel_size=random.choice([3, 5])),
            RandomOcclusion()
        ])
    
    def apply_random_transformations(self, img: np.ndarray) -> np.ndarray:
        """
        Applies a random combination of transformations to an image using the
        new compositional pipeline.
        """
        return self.default_pipeline(img)

if __name__ == "__main__":
    # Example Usage
    generator = UltimateVariantGenerator()
    
    # Create a mock board image
    mock_image = np.zeros((12 * 30, 6 * 30, 3), dtype=np.uint8) + 128
    
    print("Generating a series of augmented images...")
    
    for i in range(5):
        augmented_image = generator.apply_random_transformations(mock_image)
        cv2.imshow(f"Augmented Image {i+1}", augmented_image)
        cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    print("Generation complete.")