import torch
import torchvision

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Check torchvision version
print(f"torchvision version: {torchvision.__version__}")

# Check if CUDA (GPU) is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

# If CUDA is available, print GPU details
if cuda_available:
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Capability: {torch.cuda.get_device_capability(0)}")
else:
    print("Running on CPU.")

