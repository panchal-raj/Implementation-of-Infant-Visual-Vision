from imports import *

# Fabian Model
#from models.FabianModel import get_model  

# Sohan Model  
# from models.resnet18Sohan import get_model

# Layer-Wise ResNet18
from models.layerWiseResNet18 import get_model

# Import the transformations
from transforms.visual_acuity import VisualAcuityTransform
from transforms.color_perception import ColorPerceptionTransform


# # Paths to the models
MODEL_PATHS = MODEL_PATHS2 #refer config.py for path files 


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
def get_transformation_pipeline():
    """
    Returns a function that applies the correct transformation based on image index.
    Ensures:
      - First 25 images: No transformation
      - Next 25 images: Visual acuity blur
      - Next 25 images: Color perception adjustment
      - Last 25 images: Combination of both
    """
    def transform_image(img, index):
        # Resize and convert image to RGB format
        img = img.resize((64, 64)).convert("RGB")

        if 0 <= index < 25:
            return img  # No transformation
        elif 25 <= index < 50:
            return VisualAcuityTransform(3)(img)  # Apply visual acuity blur
        elif 50 <= index < 75:
            return ColorPerceptionTransform(3)(img)  # Apply color perception transformation
        elif 75 <= index < 100:
            img = VisualAcuityTransform(3)(img)
            img = ColorPerceptionTransform(3)(img)
            return img  # Apply both transformations
        else:
            raise ValueError("Index out of range. Expected index between 0 and 99.")

    return lambda img, idx: ToTensor()(transform_image(img, idx))


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
    Save and display an enhanced RDM heatmap with clear transformation labels and separation lines.

    Args:
        rdm (ndarray): The RDM matrix to visualize.
        layer_name (str): The name of the layer being visualized.
        model_name (str): The name of the model.
        class_images (list, optional): List of images (PIL or numpy arrays) for the axes.
    """
    num_images = rdm.shape[0]  # Number of images

    # Define tick positions and labels
    tick_positions = [0, 25, 50, 75, 99]  # Show only these major ticks
    tick_labels = ["0", "25", "50", "75", "99"]

    # Define transformation group positions (center of each block)
    group_positions = [12, 37, 62, 87]  # New center positions for 4x4 grid
    group_labels = ["No Transform", "Acuity", "Color", "All Transforms"]  


    fig, ax = plt.subplots(figsize=(20, 20))  

    # Plot the heatmap
    cax = ax.imshow(rdm, cmap="viridis", interpolation="none")  
    cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Dissimilarity", fontsize=14)

    # Set tick labels for number of images
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=14)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=14)

    # # Set transformation labels at the center of each group
    # for pos, label in zip(group_positions, group_labels):
    #     ax.text(pos, num_images + 2, label, ha="center", va="center", fontsize=16, fontweight="bold", color="black")  # X-axis labels
    #     ax.text(-8, pos, label, ha="center", va="center", fontsize=16, fontweight="bold", color="black", rotation=90)  # Y-axis labels

    # Set transformation labels at the center of each group
    for pos, label in zip(group_positions, group_labels):
        ax.text(pos, num_images + 1, label, ha="center", va="center", fontsize=16, fontweight="bold", color="black")  # X-axis labels
        ax.text(-3, pos, label, ha="center", va="center", fontsize=16, fontweight="bold", color="black", rotation=90)  # Y-axis labels


    # Add separation lines between transformation groups
    for pos in [25, 50, 75]:  # Avoid line at 0 and 99
        ax.axhline(pos - 0.5, color='white', linestyle='--', linewidth=2)
        ax.axvline(pos - 0.5, color='white', linestyle='--', linewidth=2)

    plt.title(f"RDM - {model_name} - {layer_name}\n(Images grouped by transformation type)", fontsize=18)

    # Save figure
    output_dir = "./RDMs_Figure/1.LayerWiseResNet18/with_transforms"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{model_name}_{layer_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save RDM as a .npy file
    output_dir2 = "./RDMs_npyFormat/1.LayerWiseResNet18/with_transforms"
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
    transform_pipeline = get_transformation_pipeline()
    # Apply transformations while ensuring the order
    transformed_images = torch.stack([transform_pipeline(img["image"], i) for i, img in enumerate(sampled_images)])

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
