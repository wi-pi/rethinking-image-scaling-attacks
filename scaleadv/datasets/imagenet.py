from typing import Callable, Optional

import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from scaleadv.datasets.utils import ImageFolderWithIndex

IMAGENET_PATH = 'static/datasets/imagenet/val/'
IMAGENET_NUM_CLASSES = 1000
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
])
IMAGENET_TRANSFORM_NOCROP = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])


def create_dataset(root: str = IMAGENET_PATH, transform: Optional[Callable] = IMAGENET_TRANSFORM, index: bool = False):
    cls = ImageFolderWithIndex if index is True else ImageFolder
    return cls(root, transform=transform)
