from imports import *

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output directories for storing activations and visualizations
os.makedirs("raw/data", exist_ok=True)
os.makedirs("raw/visualize", exist_ok=True)

# Define model weight paths
model_paths = MODEL_PATHS2  # Refer config.py for path files {MODEL_PATHS2 are for Layer-Wise ResNet18}

# Define layers for activation extraction
selected_layers = {
    "early": "relu",  # Early layer activation after ReLU
    "intermediate": "layer2.1.relu",  # Example intermediate layer activation
    "final": "layer4.1.relu"  # Example final layer activation
}

# Function to register forward hooks for capturing activations
def get_activation(name, activations):
    """Hook function to store activations during forward pass."""
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Load Tiny ImageNet validation set and apply transformations
def load_tiny_imagenet_data(split="valid"):
    """Load the Tiny ImageNet dataset from Hugging Face and return the specified split."""
    dataset = load_dataset("zh-plus/tiny-imagenet", split=split)
    return dataset

# Load the dataset and take one image (you can change the index to load different images)
tiny_imagenet_data = load_tiny_imagenet_data()

# Take the first image and apply the transform
image_data = tiny_imagenet_data[7]["image"]  # Assuming the image key is 'image'

# Save the original image separately
original_image_path = "raw/visualize/original_image.png"
image_data_resized = image_data.resize((64, 64))  # Resize to match feature maps
image_data_resized.save(original_image_path)  # Save the resized image
print(f"Original image saved at {original_image_path}")

# Apply the transformation
image_tensor = transform(image_data)

# Move the image to the appropriate device (GPU or CPU)
image_tensor = image_tensor.to(device)

# Track layers that have been hooked
hooked_layers = set()

# Process each model in the list of model paths
for model_name, model_path in model_paths.items():
    print(f"Processing {model_name}...")

    # Load the model and move it to the appropriate device
    model = get_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Store activations for this model
    activations = {}

    # Register hooks only once for each layer
    for layer_name in selected_layers.values():
        if layer_name not in hooked_layers:  # Prevent re-registering hooks
            model.get_submodule(layer_name).register_forward_hook(get_activation(layer_name, activations))
            hooked_layers.add(layer_name)  # Mark the layer as hooked

    # Create an HDF5 file to store feature map activations
    h5_path = f"raw/data/{model_name}.h5"
    with h5py.File(h5_path, "w") as h5f:
        # Forward pass through the model with the image tensor
        with torch.no_grad():  # Disable gradient computation for inference
            model(image_tensor.unsqueeze(0))  # Add batch dimension

        # Save activations in the HDF5 file
        for layer_name, activation in activations.items():
            data = activation.cpu().numpy()
            h5f.create_dataset(f"image_0/{layer_name}", data=data)
            print(f"Saved Activation: {layer_name} → Shape: {data.shape} in HDF5")

    # Visualization of feature maps
    visualize_dir = f"raw/visualize/{model_name}"
    os.makedirs(visualize_dir, exist_ok=True)
    for layer_name, activation in activations.items():
        activation_np = activation[0].cpu().numpy()  # Get activation for the first image
        num_features = activation_np.shape[0]  # Number of feature maps
        num_cols = 16  # Number of columns in the visualization grid
        num_rows = max(1, num_features // num_cols)  # Calculate required rows

        fig = plt.figure(figsize=(20, 20))
        grid = ImageGrid(fig, 111, nrows_ncols=(num_rows, num_cols), axes_pad=0.1)

        for ax, feature_map in zip(grid, activation_np):
            ax.imshow(feature_map, cmap="viridis")  # Apply colormap
            ax.axis("off")  # Hide axes

        # Save visualization to file
        plt.savefig(f"{visualize_dir}/{layer_name}.png")
        plt.close()

    print(f"Finished processing {model_name}!")

print("All models processed successfully!")
