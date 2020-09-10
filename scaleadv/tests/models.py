import torch.nn as nn
from torch.utils.data import DataLoader
from art.estimators.classification import PyTorchClassifier

from scaleadv.models.resnet import create_network
from scaleadv.models.layers import NormalizationLayer
from scaleadv.datasets.imagenet import create_dataset, IMAGENET_NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD


def get_classifier():
    network = create_network(IMAGENET_NUM_CLASSES, pretrained=True)
    normalize = NormalizationLayer(IMAGENET_MEAN, IMAGENET_STD)
    model = nn.Sequential(normalize, network)

    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224),
        nb_classes=IMAGENET_NUM_CLASSES,
        clip_values=(0., 1.)
    )
    return classifier


if __name__ == '__main__':
    data = create_dataset('static/datasets/imagenet/val/')
    loader = DataLoader(data, batch_size=64, num_workers=8)
    classifier = get_classifier()

    good, total = 0., 0.
    for i, (x, y) in enumerate(loader):
        y_pred = classifier.predict(x).argmax(1)
        good += (y.numpy() == y_pred).sum()
        total += x.shape[0]
        print(f'{i+1}/{len(loader)} {good/total:.3%}', end='\r')
    print()
