import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
<<<<<<< HEAD
import random
from datasets import load_dataset
=======
>>>>>>> b8baac1778cd7cc1a888eed4b3c197c80d22747c
from torchvision.transforms import Compose, Resize, ToTensor
from InfantVisualPerception.models.FabianModel import get_model
from scipy.spatial.distance import pdist, squareform
from datasets import load_dataset
import random
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# Paths to the models
MODEL_PATHS = {
    "color_perception": "./output/models/Sohan/resnet18_color_perception_final.pth",
    "curriculum": "./output/models/Sohan/resnet18_curriculum_final.pth",
    "no_curriculum": "./output/models/Sohan/resnet18_no_curriculum_final.pth",
    "visual_acuity": "./output/models/Sohan/resnet18_visual_acuity_final.pth"
}

# Load Tiny ImageNet validation set
def load_tiny_imagenet_data(split="valid"):
    """
    Load the Tiny ImageNet dataset from Hugging Face and return the specified split.
    """
    data = load_dataset("zh-plus/tiny-imagenet")
    return data[split]

# Sample random images from the validation set
def sample_random_images(data, num_samples=10):
    """
    Randomly sample a specified number of images from the dataset.

# Randomly sample 10 images from the validation set
NUM_IMAGES = 10
random.seed(42)  # Ensure reproducibility
sampled_indices = random.sample(range(len(val_data)), NUM_IMAGES)
sampled_images = val_data.select(sampled_indices)

    Returns:
        List[Image]: A list of sampled images.
    """
    sampled_indices = random.sample(range(len(data)), num_samples)
    return [data[i] for i in sampled_indices]

# Transformation pipeline for Tiny ImageNet images
def no_transform():
    """
    Transformation pipeline for Tiny ImageNet images.
    Resizes images to 64x64 and converts them to tensors.
    """
    return Compose([Resize((64, 64)), ToTensor()])

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

    layers = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]
    for layer in layers:
        layer_module = dict(model.named_modules())[layer]
        hooks.append(layer_module.register_forward_hook(register_hook(layer)))

    # Pass the images through the model
    with torch.no_grad():
        model(images)

    # Remove the hooks
    for hook in hooks:
        hook.remove()

    return feature_maps

# Calculate Representational Dissimilarity Matrix (RDM) from feature maps
def calculate_rdm(feature_map):
    flattened = feature_map.reshape(feature_map.shape[0], -1)  # Flatten feature maps
    distances = pdist(flattened, metric="correlation")
    return squareform(distances)

# Generate and save RDM heatmaps and matrices
def save_rdm(rdm, layer_name, model_name, labels):
    # Save heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(rdm, cmap="viridis")
    plt.colorbar()
    plt.title(f"RDM - {model_name} - {layer_name} - No Transform")
    # Add labels to axes
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90, fontsize=8)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=8)

    os.makedirs("./output/rdm_heatmaps/Model1", exist_ok=True)
    plt.savefig(f"./output/rdm_heatmaps/Model1/{model_name}_{layer_name}_heatmap.png")
    plt.close()

    # Save RDM as a .npy file
    os.makedirs("./output/rdm/Model1", exist_ok=True)
    np.save(f"./output/rdm/Model1/{model_name}_{layer_name}_rdm.npy", rdm)

# Main script
def main():
    # Apply no transformation pipeline
    no_transform_pipeline = no_transform()
    transformed_images = torch.stack([no_transform_pipeline(img["image"]) for img in sampled_images])

    # Convert class indices to human-readable class names
    label_mapping = val_data.features["label"].int2str  # Converts index to string
    labels = [label_mapping(img["label"]) for img in sampled_images]

    # Process models
    for model_name, model in models.items():
        if model_name == "no_transformation":
            continue  # Skip comparison with itself

        print(f"Comparing {model_name} with no_transformation...")

        # Extract feature maps for the current model
        model_features = extract_feature_maps(model, transformed_images, layers_to_compare)

        for layer_name, feature_map in feature_maps.items():
            rdm = calculate_rdm(feature_map)
            save_rdm(rdm, layer_name, model_name, labels)

if __name__ == "__main__":
    main()
