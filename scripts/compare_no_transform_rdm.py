import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, ToTensor
from models.resnet18 import get_resnet18
from scipy.spatial.distance import pdist, squareform

# Paths to the models
MODEL_PATHS = {
    "color_perception": "./output/models/Sohan/resnet18_color_perception_final.pth",
    "curriculum": "./output/models/Sohan/resnet18_curriculum_final.pth",
    "no_curriculum": "./output/models/Sohan/resnet18_no_curriculum_final.pth",
    "visual_acuity": "./output/models/Sohan/resnet18_visual_acuity_final.pth"
}

# Load Tiny ImageNet validation set
def load_tiny_imagenet_data(split="valid"):
    data = load_dataset("zh-plus/tiny-imagenet")
    return data[split]

# Load the validation data
val_data = load_tiny_imagenet_data()

# Randomly sample 10 images from the validation set
NUM_IMAGES = 10
random.seed(42)  # Ensure reproducibility
sampled_indices = random.sample(range(len(val_data)), NUM_IMAGES)
sampled_images = val_data.select(sampled_indices)

# Define the no transformation pipeline
def no_transform():
    return Compose([Resize((64, 64)), ToTensor()])

# Load models
def load_model(model_path, num_classes=200):
    model = get_resnet18(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("mps")))
    model.eval()
    return model

# Load the models
models = {name: load_model(path) for name, path in MODEL_PATHS.items()}

# Extract feature maps for RDM calculation
def extract_feature_maps(model, images):
    feature_maps = {}
    hooks = []

    def register_hook(layer_name):
        def hook(module, input, output):
            feature_maps[layer_name] = output.detach().numpy()
        return hook

    layers = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]
    for layer in layers:
        layer_module = dict(model.named_modules())[layer]
        hooks.append(layer_module.register_forward_hook(register_hook(layer)))

    with torch.no_grad():
        model(images)

    for hook in hooks:
        hook.remove()

    return feature_maps

# Calculate RDM
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
        feature_maps = extract_feature_maps(model, transformed_images)

        for layer_name, feature_map in feature_maps.items():
            rdm = calculate_rdm(feature_map)
            save_rdm(rdm, layer_name, model_name, labels)

if __name__ == "__main__":
    main()
