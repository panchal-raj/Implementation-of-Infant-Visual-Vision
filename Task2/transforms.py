from imports import *

class AcuityBlur:
    def __init__(self, age_in_months):
        self.age_in_months = age_in_months

    def __call__(self, img):
        kernel_size = 15
        max_sigma = 4.0
        min_sigma = 0.1
        sigma = max(min_sigma, max_sigma - (self.age_in_months / 12) * (max_sigma - min_sigma))
        gaussian_blur = transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))
        return gaussian_blur(img)


class ContrastAdjust:
    def __init__(self, age_in_months):
        self.age_in_months = age_in_months

    def __call__(self, img):
        age_in_weeks = self.age_in_months * 4.348125
        contrast_factor = min(age_in_weeks / 500, 1)
        return F.adjust_contrast(img, contrast_factor)