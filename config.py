import torch
AGES = [3, 6, 9, 12]

## Project PART 2
EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_CLASSES = 200  # Tiny ImageNet has 200 classes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")