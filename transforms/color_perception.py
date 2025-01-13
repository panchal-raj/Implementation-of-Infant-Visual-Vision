from PIL import ImageEnhance, Image
import numpy as np
from torchvision.transforms import functional as F

class ColorPerceptionTransform:
    def __init__(self, age_in_months):
        self.age_in_months = age_in_months

        # Red-green sensitivity (develops fully by 3 months)
        self.red_green_sensitivity = 1.0 if age_in_months >= 3 else max(0.1, (age_in_months - 2) / 1)

        # Blue-yellow sensitivity (develops fully between 4-10 months)
        self.blue_yellow_sensitivity = 0.0 if age_in_months < 4 else min(1.0, (age_in_months - 4) / 6)

        # Saturation adjustment (develops fully by 12 months)
        self.saturation_factor = min(1.0, age_in_months / 12)

    def __call__(self, image):
        # Convert image to NumPy array
        np_image = np.array(image).astype(np.float32)

        # Handle grayscale images
        if len(np_image.shape) == 2:  # Grayscale image
            np_image = np.expand_dims(np_image, axis=-1)  # Add channel dimension
            np_image = np.repeat(np_image, 3, axis=-1)    # Convert to pseudo-RGB

        # Calculate original brightness per pixel
        original_brightness = np_image.mean(axis=2, keepdims=True)

        if self.age_in_months <= 1:
            # Grayscale-like appearance for newborns
            luminance = np_image.mean(axis=2, keepdims=True)
            np_image = np.repeat(luminance, 3, axis=2)
        else:
            # Apply red-green sensitivity
            np_image[:, :, 0] *= self.red_green_sensitivity  # Red channel
            np_image[:, :, 1] *= self.red_green_sensitivity  # Green channel

            # Apply blue-yellow sensitivity
            np_image[:, :, 2] *= self.blue_yellow_sensitivity  # Blue channel

        # Normalize brightness to match the original
        adjusted_brightness = np_image.mean(axis=2, keepdims=True)
        brightness_ratio = original_brightness / np.clip(adjusted_brightness, 1e-6, np.max(original_brightness))
        np_image *= brightness_ratio

        # Clip values to valid range
        np_image = np.clip(np_image, 0, 255)

        # Convert back to PIL Image
        adjusted_image = Image.fromarray(np_image.astype(np.uint8))

        # Enhance saturation for age-based perception
        enhancer = ImageEnhance.Color(adjusted_image)
        enhanced_image = enhancer.enhance(self.saturation_factor)

        return enhanced_image


class ContrastTransform:
    def __init__(self, age_in_months):
        self.age_in_months = age_in_months

    def __call__(self, img):
        # Convert age in months to age in weeks
        age_in_weeks = self.age_in_months * (365.25 / 12) / 7

        # Calculate contrast factor (develops fully by 12 months)
        contrast_factor = min(age_in_weeks / 52, 1.0) if self.age_in_months <= 12 else 1.0

        # Apply contrast adjustment
        return F.adjust_contrast(img, contrast_factor)
