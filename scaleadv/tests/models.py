import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescentPyTorch

from scaleadv.models.resnet import create_network
from scaleadv.models.layers import NormalizationLayer
from scaleadv.datasets.imagenet import create_dataset, IMAGENET_NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD
from scaleadv.attacks.adv import AdvAttack


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

def evaluate(data_loader, classifier, caption='Test', attack=None):
    good, total = 0., 0.
    for i, (x, y) in enumerate(data_loader):
        if attack is not None:
            x = attack.generate(x)
        y_pred = classifier.predict(x).argmax(1)
        good += (y.numpy() == y_pred).sum()
        total += x.shape[0]
        print(f'{caption}: {i+1}/{len(loader)} {good/total:.3%}', end='\r')
    print()


if __name__ == '__main__':
    data = create_dataset('static/datasets/imagenet/val/')
    loader = DataLoader(data, batch_size=128, num_workers=8)
    classifier = get_classifier()
    art = ProjectedGradientDescentPyTorch(classifier, norm=np.inf, eps=0.3, eps_step=0.02, max_iter=20)
    attack = AdvAttack(art)
    evaluate(loader, classifier, 'Good')
    evaluate(loader, classifier, 'Evil', attack)

