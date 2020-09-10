import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

IMAGENET_NUM_CLASSES = 1000
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
])

def create_dataset(root):
    return ImageFolder(root, transform=IMAGENET_TRANSFORM)
