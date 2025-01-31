import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision import transforms
from scipy.spatial.distance import pdist, squareform
from datasets import load_dataset
import random
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import h5py
from mpl_toolkits.axes_grid1 import ImageGrid
from models.layerWiseResNet18 import get_model  # Import your custom model --> Layer-Wise ResNet18
# from models.FabianModel import get_model
from config import MODEL_PATHS1, MODEL_PATHS2