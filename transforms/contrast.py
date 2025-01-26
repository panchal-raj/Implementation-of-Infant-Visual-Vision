from torchvision.transforms import functional as F

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