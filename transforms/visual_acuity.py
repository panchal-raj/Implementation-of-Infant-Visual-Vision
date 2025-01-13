from PIL import ImageFilter
import math


class VisualAcuityTransform:
    def __init__(self, age_in_months):
        self.age_in_months = age_in_months

        # Define the maximum blur radius (newborns) and minimum (perfect vision).
        max_blur = 3  # Corresponds to 20/600 vision
        min_blur = 0  # Corresponds to 20/20 vision

        # Update the decay formula to reflect 12 months timeline
        # Updated linear decay formula based on the 12-month timeline described in the paper
        if age_in_months <= 12:
            self.blur_radius = max_blur - ((max_blur - min_blur) * (age_in_months / 12))
        else:
            self.blur_radius = min_blur  # Beyond 12 months, perfect vision assumed

    def __call__(self, image):
        if self.blur_radius <= 0:
            return image
        return image.filter(ImageFilter.GaussianBlur(self.blur_radius))
