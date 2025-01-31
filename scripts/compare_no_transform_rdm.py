import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor
from models.FabianModel import get_model
from scipy.spatial.distance import pdist, squareform
from datasets import load_dataset
import random
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# Paths to the models
MODEL_PATHS = {
    "visual_acuity": "InfantVisualPerception/weights/FabianResNet/resnet18_tinyimagenet_acuity.pth",
    "contrast_adjust": "InfantVisualPerception/weights/FabianResNet/resnet18_tinyimagenet_contrast.pth",
    "both_transforms": "InfantVisualPerception/weights/FabianResNet/resnet18_tinyimagenet_both.pth",
    "no_transformation": "InfantVisualPerception/weights/FabianResNet/resnet18_tinyimagenet_default.pth"
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


# Use AxesImage to draw the thumbnails on the heatmap's axes:
def add_images_to_axes(ax, images, positions, axis="x"):
    """
    Add images as thumbnails to the heatmap axes.

    Args:
        ax (matplotlib.axes.Axes): The axes of the heatmap.
        images (list): List of images as PIL or numpy arrays.
        positions (list): Positions (ticks) where images should appear.
        axis (str): Whether to add images to the "x" or "y" axis.
        size (float): The relative size of the thumbnails.
    """
    num_images = len(images)  # Total number of images
    size = 1.0 / num_images  # Size relative to the number of images (RDM cells)
    
    for pos, img in zip(positions, images):
        imagebox = OffsetImage(img, zoom=size)
        if axis == "x":
            ab = AnnotationBbox(imagebox, (pos, -0.5), frameon=False, box_alignment=(0.5, 0))
        else:
            ab = AnnotationBbox(imagebox, (-0.5, pos), frameon=False, box_alignment=(0, 0.5))
        ax.add_artist(ab)

# Save the RDM as a heatmap image
def save_rdm_heatmap(rdm, layer_name, model_name, class_images=None):
    """
    Save and optionally display an enhanced RDM heatmap.

    Args:
        rdm (ndarray): The RDM matrix to visualize.
        layer_name (str): The name of the layer being visualized.
        model_name (str): The name of the model.
        class_images (list, optional): List of images (PIL or numpy arrays) for the axes.
    """
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    #plot the heatmap
    plt.imshow(rdm, cmap="viridis")
    plt.colorbar(label="Correlation Distance")

    # Annotate each cell with its value
    for i in range(rdm.shape[0]):
        for j in range(rdm.shape[1]):
            plt.text(j, i, f"{rdm[i, j]:.2f}", ha="center", va="center", fontsize=12, color="white")

    # Add images to axes if provided
    if class_images:
        ticks = np.arange(len(class_images))
        add_images_to_axes(ax, class_images, ticks, axis="x")  # Add images to x-axis
        add_images_to_axes(ax, class_images, ticks, axis="y")  # Add images to y-axis

    # Configure tick positions and hide text labels
    ax.set_xticks(np.arange(len(class_images)))
    ax.set_yticks(np.arange(len(class_images)))
    ax.tick_params(axis="x", labelbottom=False, bottom=False)
    ax.tick_params(axis="y", labelleft=False, left=False)

    # # Set axis labels
    # if class_labels:
    #     plt.xticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=90, fontsize=8)
    #     plt.yticks(ticks=np.arange(len(class_labels)), labels=class_labels, fontsize=8)
    # else:
    #     plt.xticks(fontsize=8)
    #     plt.yticks(fontsize=8)

    plt.title(f"RDM - {model_name} - {layer_name}", fontsize=12)
    output_dir = "./output7/rdm_heatmaps_FabianModel/"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{model_name}_{layer_name}.png")
    plt.close()



# Main function to compute RDMs
def main():
    # Layers to compare: Initial, intermediate, and final layers, including conv1 and fc
    layers_to_compare = ["conv1", "layer1", "layer3", "layer4", "fc"]

    # Load Tiny ImageNet validation set
    val_data = load_tiny_imagenet_data(split="valid")

    # Randomly sample 100 images from the validation set
    NUM_IMAGES = 10
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