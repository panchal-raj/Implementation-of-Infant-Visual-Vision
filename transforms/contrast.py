from torchvision.transforms import functional as F

class ContrastTransform:
    def __init__(self, age_in_months):
        self.age_in_months = age_in_months

    def __call__(self, img):
        if self.age_in_months < 2:
            contrast_factor = 0.2
        else:
            # Gradual increase after 2 months
            age_in_weeks = self.age_in_months * (365.25 / 12) / 7
            contrast_factor = max(0.2, min(age_in_weeks / 52, 1.0))
        return F.adjust_contrast(img, contrast_factor)