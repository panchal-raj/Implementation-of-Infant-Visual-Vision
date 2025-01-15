# scripts/compare_models.py

import os
import torch
import brainscore
from brainscore.benchmarks import BenchmarkPool
from brainscore.interface import BrainModel
from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from torch import nn
from src.dataloader import create_no_transform, PreprocessedDataset
from config import DEVICE, MODEL_OUTPUT_DIR, AGES, NUM_CLASSES

def load_model(model_path, num_classes=NUM_CLASSES):
    """Loads a ResNet18 model from a given path."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def extract_layer_activations(model, layer_name, inputs):
    """Extract activations from a specific layer."""
    activations = []

    def hook(module, input, output):
        activations.append(output.cpu().detach().numpy())

    layer = dict([*model.named_modules()])[layer_name]
    hook_handle = layer.register_forward_hook(hook)
    _ = model(inputs.to(DEVICE))
    hook_handle.remove()
    return np.concatenate(activations, axis=0)

def compute_rdm(activations):
    """Computes the Representational Dissimilarity Matrix (RDM)."""
    return squareform(pdist(activations, metric="euclidean"))

def plot_rdm(rdm, title, output_path):
    """Plots and saves the RDM as a heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(rdm, cmap='viridis')
    plt.colorbar(label="Dissimilarity")
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def compare_with_brainscore(models):
    """Compare models with Brain-Score benchmark."""
    pool = BenchmarkPool()
    results = {}
    for model_name, model in models.items():
        brainscore_model = BrainModel(model)
        for benchmark in pool:
            score = pool[benchmark](brainscore_model)
            print(f"Model: {model_name}, Benchmark: {benchmark}, Score: {score}")
            results[(model_name, benchmark)] = score
    return results

def main():
    model_paths = {
        "color_perception": os.path.join(MODEL_OUTPUT_DIR, "Submitted", "resnet18_color_perception_final.pth"),
        "curriculum": os.path.join(MODEL_OUTPUT_DIR, "Submitted", "resnet18_curriculum_final.pth"),
        "no_curriculum": os.path.join(MODEL_OUTPUT_DIR, "Submitted", "resnet18_no_curriculum_final.pth"),
        "visual_acuity": os.path.join(MODEL_OUTPUT_DIR, "Submitted", "resnet18_visual_acuity_final.pth"),
    }

    # Load models
    models = {name: load_model(path) for name, path in model_paths.items()}

    # Define dataset and transformations
    transform = create_no_transform()
    dataset_path = "./datasets/your_dataset"  # Update with your dataset path
    dataset = PreprocessedDataset(dataset_path, transform=transform)

    # Select a subset of 100 images for RDM computation
    subset = torch.utils.data.Subset(dataset, list(range(100)))
    dataloader = torch.utils.data.DataLoader(subset, batch_size=16, shuffle=False, num_workers=4)

    images = []
    for img, _ in dataloader:
        images.append(img)
    inputs = torch.cat(images, dim=0)

    # Compute RDMs
    layers = ["layer2.0.conv1", "layer3.0.conv1", "layer4.0.conv1"]  # Specify layers of interest
    for model_name, model in models.items():
        for layer in layers:
            activations = extract_layer_activations(model, layer, inputs)
            rdm = compute_rdm(activations)
            output_path = f"output/figure/{model_name}_{layer}_rdm.png"
            plot_rdm(rdm, f"RDM for {model_name} at {layer}", output_path)
            print(f"Saved RDM for {model_name} at {layer} to {output_path}")

    # Compare with Brain-Score benchmarks
    brain_score_results = compare_with_brainscore(models)
    print("Brain-Score Results:", brain_score_results)

if __name__ == "__main__":
    main()
