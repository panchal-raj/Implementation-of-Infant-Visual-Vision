import os
from PIL import Image
import matplotlib.pyplot as plt
from src.dataloader import create_acuity_transform, create_color_transform, create_contrast_transform, create_age_based_transform
from datasets import load_dataset
from torchvision import transforms

# Load Tiny ImageNet sample image from Hugging Face
def load_tinyimagenet_data():
    data = load_dataset("zh-plus/tiny-imagenet")
    return data['train']

# Save image function
def save_image(image, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    image.save(path)

# Plot and save images with different transformations applied
def plot_and_save_transformed_images(sample_image, ages, output_dir):
    for age in ages:
        # Apply individual transforms
        acuity_transform = create_acuity_transform(age)
        color_transform = create_color_transform(age)
        contrast_transform = create_contrast_transform(age)
        combined_transform = create_age_based_transform(age)

        # Apply and save transformed images
        acuity_image = acuity_transform(sample_image)
        color_image = color_transform(sample_image)
        contrast_image = contrast_transform(sample_image)
        combined_image = combined_transform(sample_image)

        # Convert tensors to PIL images (if necessary)
        acuity_image = transforms.ToPILImage()(acuity_image) if not isinstance(acuity_image, Image.Image) else acuity_image
        color_image = transforms.ToPILImage()(color_image) if not isinstance(color_image, Image.Image) else color_image
        contrast_image = transforms.ToPILImage()(contrast_image) if not isinstance(contrast_image, Image.Image) else contrast_image
        combined_image = transforms.ToPILImage()(combined_image) if not isinstance(combined_image, Image.Image) else combined_image

        # Create a figure with all images
        fig, axs = plt.subplots(1, 5, figsize=(25, 5))
        axs[0].imshow(sample_image)
        axs[0].set_title("Original")
        axs[1].imshow(acuity_image)
        axs[1].set_title("Acuity")
        axs[2].imshow(color_image)
        axs[2].set_title("Color")
        axs[3].imshow(contrast_image)
        axs[3].set_title("Contrast")
        axs[4].imshow(combined_image)
        axs[4].set_title("Combined")

        # Remove axes
        for ax in axs:
            ax.axis('off')

        # Save the plot
        save_path = os.path.join(output_dir, f"age_{age}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    sample_image = load_tinyimagenet_data()[0]['image']
    ages = [1, 3, 6, 9, 12]
    output_dir="output/transformed_images"
    os.makedirs(output_dir, exist_ok=True)
    plot_and_save_transformed_images(sample_image, ages, output_dir)
