import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor

from scipy.spatial.distance import pdist, squareform
from datasets import load_dataset
import random
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
# from config import MODEL_PATHS

from models.FabianModel import get_model
#from models.resnet18Sohan import get_model


# # Paths to the models
# MODEL_PATHS = MODEL_PATHS #refer config.py for path files 

MODEL_PATHS = {
    "visual_acuity": "./weights/SohanResNet/resnet18_visual_acuity_final.pth",
    "color_adjust": "./weights/SohanResNet/resnet18_color_perception_final.pth",
    "both_transforms": "./weights/SohanResNet/resnet18_curriculum_final.pth",
    "no_transformation": "./weights/SohanResNet/resnet18_no_curriculum_final.pth"
}


# Load Tiny ImageNet validation set
def load_tiny_imagenet_data(split="valid"):
    """
    Load the Tiny ImageNet dataset from Hugging Face and return the specified split.
    """
    data = load_dataset("zh-plus/tiny-imagenet")
    return data[split]

# Sample random images from the validation set
def sample_random_images(data, num_samples=100):
    """
    Randomly sample a specified number of images from the dataset.

    Args:
        data (Dataset): The Tiny ImageNet dataset split.
        num_samples (int): Number of images to sample.

    Returns:
        List[Image]: A list of sampled images.
    """
    sampled_indices = random.sample(range(len(data)), num_samples)
    return [data[i] for i in sampled_indices]

# Transformation pipeline for Tiny ImageNet images
def no_transform():
    """
    Ensure all images are resized to (64, 64) and converted to 3-channel RGB.
    """
    return Compose([
        Resize((64, 64)), 
        lambda img: img.convert("RGB"),  # Ensure RGB mode (3 channels)
        ToTensor()
    ])

# Load the model from a given path
def load_model(model_path, num_classes=200):
    """
    Load a ResNet18 model with weights from the specified path.

    Args:
        model_path (str): Path to the model weights.
        num_classes (int): Number of output classes for the model.

    Returns:
        torch.nn.Module: The loaded model in evaluation mode.
    """
    model = get_model(num_classes=num_classes)  # Initialize model
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))  # Load weights
    model.load_state_dict(state_dict)  # Apply weights
    model.eval()  # Set the model to evaluation mode
    return model

# Extract feature maps from specific layers of the model
def extract_feature_maps(model, images, layers_to_extract):
    feature_maps = {}
    hooks = []

    # Register hooks for the specified layers
    def register_hook(layer_name):
        def hook(module, input, output):
            feature_maps[layer_name] = output.detach().numpy()
        return hook

    for layer_name in layers_to_extract:
        layer_module = dict(model.named_modules())[layer_name]
        hooks.append(layer_module.register_forward_hook(register_hook(layer_name)))

    # Pass the images through the model
    with torch.no_grad():
        model(images)

    # Remove the hooks
    for hook in hooks:
        hook.remove()

    return feature_maps

# Calculate Representational Dissimilarity Matrix (RDM) from feature maps
def calculate_rdm(feature_map):
    flattened = feature_map.reshape(feature_map.shape[0], -1)  # Flatten the feature maps
    distances = pdist(flattened, metric="correlation")  # Compute pairwise correlation distances
    return squareform(distances)  # Convert to square matrix


# Save the RDM as a heatmap image and .npy format
def save_rdm_heatmap(rdm, layer_name, model_name, class_images=None):
    """
    Save and display an enhanced RDM heatmap.

    Args:
        rdm (ndarray): The RDM matrix to visualize.
        layer_name (str): The name of the layer being visualized.
        model_name (str): The name of the model.
        class_images (list, optional): List of images (PIL or numpy arrays) for the axes.
    """
    num_images = rdm.shape[0]  # Number of images

    # Increased figure size for better visibility
    fig, ax = plt.subplots(figsize=(20, 20))  

    # Plot the heatmap with better settings
    cax = ax.imshow(rdm, cmap="viridis", interpolation="none")  
    cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Dissimilarity", fontsize=14)

    # Set ticks for all images
    # ax.set_xticks(np.arange(num_images))
    ax.set_yticks(np.arange(num_images))

    plt.title(f"RDM - {model_name} - {layer_name}", fontsize=18)
    
    # Save figure
    output_dir = "./RDMs_Figure/2.SohanResNet/no_transforms"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{model_name}_{layer_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save RDM as a .npy file
    output_dir2 = "./RDMs_npyFormat/2.SohanResNet/no_transforms"
    os.makedirs(output_dir2, exist_ok=True)
    np.save(f"{output_dir2}/{model_name}_{layer_name}_rdm.npy", rdm)


# Main function to compute RDMs
def main():
    # Layers to compare: Initial, intermediate, and final layers, including conv1 and fc
    layers_to_compare = ["conv1", "layer1", "layer3", "layer4", "fc"]

    # Load Tiny ImageNet validation set
    val_data = load_tiny_imagenet_data(split="valid")

    # Randomly sample 100 images from the validation set
    NUM_IMAGES = 100
    sampled_images = sample_random_images(val_data, NUM_IMAGES)

    # Convert PIL images to numpy arrays for display on heatmaps
    # class_images = [Image.open(img["image"]).resize((64, 64)) for img in sampled_images]
    class_images = [img["image"].resize((64, 64)) if isinstance(img["image"], Image.Image) else Image.open(img["image"]).resize((64, 64))
                    for img in sampled_images]
    # Apply the transformation pipeline to images
    transform_pipeline = no_transform()
    transformed_images = torch.stack([transform_pipeline(img["image"]) for img in sampled_images])

    # Load all models
    models = {name: load_model(path) for name, path in MODEL_PATHS.items()}

    # Compare each model with the no-transformation baseline
    baseline_model = models["no_transformation"]
    baseline_features = extract_feature_maps(baseline_model, transformed_images, layers_to_compare)

    # Dummy class labels for demonstration
    class_labels = [img["label"] for img in sampled_images]

    for model_name, model in models.items():
        if model_name == "no_transformation":
            continue  # Skip comparison with itself

        print(f"Comparing {model_name} with no_transformation...")

        # Extract feature maps for the current model
        model_features = extract_feature_maps(model, transformed_images, layers_to_compare)

        # Compute and save RDMs for each layer
        for layer_name in layers_to_compare:
            rdm = calculate_rdm(model_features[layer_name] - baseline_features[layer_name])
            save_rdm_heatmap(rdm, layer_name, model_name, class_images)

if __name__ == "__main__":
    main()
