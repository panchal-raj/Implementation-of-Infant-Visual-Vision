# Import common libraries
import numpy as np
import pandas as pd

import random
import os
import time
import copy

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import functional as F, ToPILImage
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split, Subset
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the directory
train_dir = "/kaggle/input/tiny-imagenet/tiny-imagenet-200/train"

age_groups = [1, 6, 12, 24] # DevelopmentalCurriculum
batch_size = 64

# Load the entire dataset without splitting
base_dataset = torchvision.datasets.ImageFolder(train_dir)

# Hyperparameters
num_epochs = 128