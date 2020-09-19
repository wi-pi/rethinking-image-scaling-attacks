import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from art.attacks.attack import EvasionAttack
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier
from torchvision.models import resnet50

from scaleadv.datasets.imagenet import create_dataset, IMAGENET_MEAN, IMAGENET_STD
from scaleadv.models.layers import NormalizationLayer


def _to_batch(x: np.ndarray):
    # [0, 255](W, H, C) to [0, 1](1, C, W, H) in float32
    x = x.astype(np.float32) / 255.
    x = x.transpose((2, 0, 1))[None, ...]
    return x


def _to_single(x: np.ndarray):
    # [0, 1](1, C, W, H) to [0, 255](W, H, C) in uint8
    x = x[0].transpose((1, 2, 0)) * 255.
    x = x.round().astype(np.uint8)
    return x


class AdvAttack(object):

    def __init__(self, classifier: PyTorchClassifier, attacker: EvasionAttack):
        assert isinstance(classifier, PyTorchClassifier)
        assert isinstance(attacker, EvasionAttack)
        self.classifier = classifier
        self.attacker = attacker

    def predict(self, x: np.ndarray):
        self._validate(x)
        x = _to_batch(x)
        pred = self.classifier.predict(x)
        return pred.argmax(1)[0], pred

    def generate(self, x: np.ndarray, y: int):
        self._validate(x)
        x = _to_batch(x)
        y = np.array([y])
        x_adv = self.attacker.generate(x, y)
        return _to_single(x_adv)

    def _validate(self, x: np.ndarray):
        assert x.dtype == np.uint8
        assert x.ndim == 3
        assert x.shape[::-1] == self.classifier.input_shape


def test():
    # load data
    dataset = create_dataset(transform=T.Resize((224, 224)))
    _, x, y = dataset[1000]
    x = np.array(x)

    # load classifier
    model = nn.Sequential(
        NormalizationLayer(IMAGENET_MEAN, IMAGENET_STD),
        resnet50(pretrained=True)
    ).eval()
    classifier = PyTorchClassifier(model, nn.CrossEntropyLoss(), (3, 224, 224), 1000, clip_values=(0., 1.))

    # load attacker
    # TODO: Support targeted attack.
    attacker = ProjectedGradientDescentPyTorch(classifier, np.inf, 0.03, 0.007, max_iter=10)
    aa = AdvAttack(classifier, attacker)

    # evaluate
    x_evil = aa.generate(x, y)
    y_pred, _ = aa.predict(x)
    y_evil, _ = aa.predict(x_evil)
    print('True', y)
    print('Pred', y_pred)
    print('Evil', y_evil)


if __name__ == '__main__':
    test()
