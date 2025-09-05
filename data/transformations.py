from torchvision.transforms.transforms import Resize
from torchvision.transforms import transforms
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import random
import numpy as np
import torch

# Define data transformation


transform_train = transforms.Compose([
    transforms.Lambda(lambda x: x[:, 65:, :]), 
    transforms.ToPILImage(), 
    transforms.Resize((224, 224), antialias=True), 
    torchvision.transforms.AugMix(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])


transform_test = transforms.Compose([
    transforms.Lambda(lambda x: x[:, 65:, :]),
    transforms.ToPILImage(),
    transforms.Resize((224, 224), antialias=True), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

