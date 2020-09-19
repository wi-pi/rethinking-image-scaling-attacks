import torchvision.transforms as T
from scaleadv.datasets.utils import ImageFolderWithIndex

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


def create_dataset(root, transform=IMAGENET_TRANSFORM):
    return ImageFolderWithIndex(root, transform=transform)
