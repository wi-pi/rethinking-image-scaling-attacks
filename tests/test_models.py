from typing import Optional

import numpy as np
import pytest
import torch.nn as nn
import torchvision.transforms as T
from art.attacks import EvasionAttack
from art.attacks.evasion import ProjectedGradientDescentPyTorch as PGD
from art.estimators.classification import PyTorchClassifier
from loguru import logger
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from scaleadv.datasets import get_imagenet
from scaleadv.models import resnet50
from scaleadv.models.layers import NormalizationLayer


def _get_data_loader(batch_size, num_workers):
    # trans = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    trans = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    dataset = get_imagenet(split='val', transform=trans)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader


def _evaluate(loader: DataLoader, classifier: PyTorchClassifier, attack: Optional[EvasionAttack], desc: str):
    acc = []
    for x, y in tqdm(loader, desc=desc):
        if attack is not None:
            x = attack.generate(x, y)
        y_pred = classifier.predict(x).argmax(1)
        acc.append(y.numpy() == y_pred)
    acc = np.mean(np.concatenate(acc))
    return acc


class TestModelPerformance(object):
    """
    Expected Performance
      * `T.Compose([T.Resize((224, 224)), T.ToTensor()])`
            none    2
        acc 0.74548 0.53628
        rob 0.00274 0.29964
      * `T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])`
            none    2
        acc 0.76130 0.57900
        rob 0.00370 0.35158
    """
    batch_size = 256
    loader = _get_data_loader(batch_size=batch_size, num_workers=16)
    model_list = ['none', '2']
    acc_list = [0.76130, 0.57900]
    rob_list = [0.00370, 0.35158]

    @pytest.mark.parametrize('model,true_acc', zip(model_list, acc_list))
    def test_accuracy(self, model: str, true_acc: float):
        classifier = self._get_classifier(model)
        test_acc = _evaluate(self.loader, classifier, attack=None, desc=f'Evaluate Model "{model}"')
        logger.info(f'Model "{model}" accuracy is {test_acc:.3%}')
        assert np.isclose(test_acc, true_acc, atol=0.01), f'{test_acc} != {true_acc}'

    @pytest.mark.parametrize('model,true_rob', zip(model_list, rob_list))
    def test_robustness(self, model: str, true_rob: float):
        classifier = self._get_classifier(model)
        attack = PGD(classifier, 2, eps=3.0, eps_step=0.375, max_iter=20, batch_size=self.batch_size, verbose=False)
        test_rob = _evaluate(self.loader, classifier, attack=attack, desc=f'Evaluate Model "{model}"')
        logger.info(f'Model "{model}" robustness is {test_rob:.3%}')
        assert np.isclose(test_rob, true_rob, atol=0.1), f'{test_rob} != {true_rob}'

    @staticmethod
    def _get_classifier(model: str) -> PyTorchClassifier:
        model = nn.Sequential(NormalizationLayer.preset('imagenet'), resnet50(model)).eval()
        model = DataParallel(model).cuda()
        classifier = PyTorchClassifier(model, nn.CrossEntropyLoss(), (3, 224, 224), 1000, clip_values=(0, 1))
        return classifier
