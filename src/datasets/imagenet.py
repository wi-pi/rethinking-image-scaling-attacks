from pathlib import Path

from torchvision.datasets import ImageFolder

IMAGENET_PATH = Path('static/datasets/imagenet/')
IMAGENET_NUM_CLASSES = 1000
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_imagenet(split='val', transform=None, target_transform=None):
    root = str(IMAGENET_PATH / split)
    return ImageFolder(root, transform, target_transform)
