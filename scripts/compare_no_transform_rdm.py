import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, ToTensor
from models.resnet18 import get_resnet18
from scipy.spatial.distance import pdist, squareform

# Paths to the models
MODEL_PATHS = {
    "color_perception": "./output/models/Submitted/resnet18_color_perception_final.pth",
    "curriculum": "./output/models/Submitted/resnet18_curriculum_final.pth",
    "no_curriculum": "./output/models/Submitted/resnet18_no_curriculum_final.pth",
    "visual_acuity": "./output/models/Submitted/resnet18_visual_acuity_final.pth"
}

# Load Tiny ImageNet validation set
def load_tiny_imagenet_data(split="valid"):
    data = load_dataset("zh-plus/tiny-imagenet")
    return data[split]

# Load the validation data
val_data = load_tiny_imagenet_data()

# Sample 100 images from the validation set
NUM_IMAGES = 100
sampled_images = val_data.select(range(NUM_IMAGES))

# Define the no transformation pipeline
def no_transform():
    return Compose([Resize((64, 64)), ToTensor()])

# Load models
def load_model(model_path, num_classes=200):
    model = get_resnet18(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
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

    layers = ["layer1", "layer2", "layer3", "layer4"]
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

# Generate and save RDM heatmaps
def save_rdm_heatmap(rdm, layer_name, model_name):
    plt.figure(figsize=(10, 8))
    plt.imshow(rdm, cmap="viridis")
    plt.colorbar()
    plt.title(f"RDM - {model_name} - {layer_name} - No Transform")
    os.makedirs("./output/rdm_heatmaps/no_transform", exist_ok=True)
    plt.savefig(f"./output/rdm_heatmaps/no_transform/{model_name}_{layer_name}.png")
    plt.close()

# Main script
def main():
    no_transform_pipeline = no_transform()
    transformed_images = torch.stack([no_transform_pipeline(img["image"]) for img in sampled_images])

    for model_name, model in models.items():
        feature_maps = extract_feature_maps(model, transformed_images)

        for layer_name, feature_map in feature_maps.items():
            rdm = calculate_rdm(feature_map)
            save_rdm_heatmap(rdm, layer_name, model_name)

if __name__ == "__main__":
    main()
