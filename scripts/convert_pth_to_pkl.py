import torch
import pickle
import os

# Define the paths
input_folder = "output/models/Sohan"
output_folder = "output/models/Sohan/converted/"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# List of model files to convert
model_files = [
    "resnet18_color_perception_final.pth",
    "resnet18_curriculum_final.pth",
    "resnet18_no_curriculum_final.pth",
    "resnet18_visual_acuity_final.pth"
]

def convert_pth_to_pkl(input_path, output_path):
    """Load a .pth model and save it as a .pkl file."""
    # Load the .pth file
    model_data = torch.load(input_path, map_location=torch.device('cpu'))
    
    # Save it as a .pkl file
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Converted: {input_path} -> {output_path}")

def main():
    for model_file in model_files:
        input_path = os.path.join(input_folder, model_file)
        output_path = os.path.join(output_folder, model_file.replace('.pth', '.pkl'))
        
        # Convert the file
        convert_pth_to_pkl(input_path, output_path)

if __name__ == "__main__":
    main()
