import torch
import os
from models.resnet18Sohan import get_resnet18  # Import the custom ResNet18 model
from datasets import load_dataset  # For loading Tiny ImageNet dataset
import random  # For selecting random images
from torchvision import transforms
import json  # For saving results
from brainscore_core.benchmarks import benchmark_pool, score_model  # For Brainscore integration
from brainscore_core.model_interface import BrainModel

# Define the folder where your models are located
MODEL_FOLDER = "output/models/Sohan/"
RESULTS_FOLDER = "output/brainscore_benchmark/"

# Ensure the results folder exists
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Define your models
MODEL_FILES = [
    "resnet18_color_perception_final.pth",
    "resnet18_curriculum_final.pth",
    "resnet18_no_curriculum_final.pth",
    "resnet18_visual_acuity_final.pth"
]

def load_tiny_imagenet_data(split="valid"):
    """Load Tiny ImageNet data from Hugging Face."""
    data = load_dataset("zh-plus/tiny-imagenet")
    return data[split]

class ResNet18BrainModel(BrainModel):
    def __init__(self, model_path, identifier, num_classes=200):
        self.identifier = identifier
        self.version = "1.0"
        self.model = get_resnet18(num_classes=num_classes)

        # Load the state dictionary into the model
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def look_at(self, stimuli, layers=None):
        """Run the model on stimuli and return layer responses."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        input_data = torch.stack([transform(image) for image in stimuli])

        with torch.no_grad():
            outputs = self.model(input_data)
        return {"final_layer": outputs}

    def visual_degrees(self):
        """Return the visual degrees of the model."""
        return 8  # Adjust based on the model's receptive field

# Preprocessing for Tiny ImageNet images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load validation data from Tiny ImageNet
tiny_imagenet_data = load_tiny_imagenet_data(split="valid")

# Select 10 random images from the validation set
random_indices = random.sample(range(len(tiny_imagenet_data)), 10)
random_images = [tiny_imagenet_data[i]["image"] for i in random_indices]

# Register your models and evaluate them on Brainscore
for model_file in MODEL_FILES:
    model_path = os.path.join(MODEL_FOLDER, model_file)

    if not os.path.exists(model_path):
        print(f"Model file {model_file} not found in {MODEL_FOLDER}. Skipping...")
        continue

    # Create a Brainscore-compatible model
    model_identifier = model_file.replace('.pth', '')
    brain_model = ResNet18BrainModel(model_path, model_identifier)

    # Evaluate the model on Brainscore benchmarks
    print(f"Evaluating {model_file} on Brainscore benchmarks...")
    for benchmark_name, benchmark in benchmark_pool.items():
        try:
            score = score_model(model_identifier, brain_model, benchmark=benchmark)

            # Save results to a file
            results_path = os.path.join(RESULTS_FOLDER, f"{model_identifier}_{benchmark_name}_results.json")
            with open(results_path, 'w') as f:
                json.dump({"benchmark": benchmark_name, "score": score.raw}, f)

            print(f"Benchmark {benchmark_name} completed. Results saved to {results_path}")
        except Exception as e:
            print(f"Error evaluating {model_identifier} on {benchmark_name}: {e}")
