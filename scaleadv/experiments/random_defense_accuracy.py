import argparse
from concurrent.futures.process import ProcessPoolExecutor
from typing import List

import numpy as np
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier
from defenses.detection.fourier.FourierPeakMatrixCollector import FourierPeakMatrixCollector, PeakMatrixMethod
from defenses.prevention.PreventionDefenseGenerator import PreventionDefenseGenerator
from defenses.prevention.PreventionDefenseType import PreventionTypeDefense
from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries
from torch.nn import DataParallel
from torchvision.models import resnet50, vgg19
from tqdm import tqdm

from scaleadv.datasets.imagenet import IMAGENET_MEAN, IMAGENET_STD
from scaleadv.datasets.imagenet import create_dataset
from scaleadv.models.layers import NormalizationLayer

ARCH = {
    'res50': resnet50,
    'vgg19': vgg19
}
LIB = SuppScalingLibraries.CV
ALGO = SuppScalingAlgorithms.LINEAR


def topk_acc(pred: np.ndarray, y: int, k: int = 1):
    topk = pred.argsort()[:, -k:]
    good = np.equal(topk, y).any(axis=1).astype(np.float32)
    return good.mean()


def show(caption: str, pred: np.ndarray, y: int, k: List[int]):
    acc = [topk_acc(pred, y, i) for i in k]
    print(caption, ' '.join([f'{a:.2%}' for a in acc]))


def get(x):
    return scaling.scale_image(defense.make_image_secure(x))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--id', type=int, required=True, help='Image ID.', metavar='ID')
    p.add_argument('-r', '--repeat', type=int, required=True, help='Number of experiments.', metavar='T')
    p.add_argument('-a', '--arch', type=str, default='res50', help='Architecture of network.')
    args = p.parse_args()

    # load data to [0, 255] ndarray of (H, W, C)
    dataset = create_dataset(transform=None)
    _, x_img, y_img = dataset[args.id]
    x_src = np.array(x_img)

    # load classifier
    network = ARCH[args.arch]
    model = nn.Sequential(NormalizationLayer(IMAGENET_MEAN, IMAGENET_STD), network(pretrained=True)).eval()
    model = DataParallel(model).cuda()
    classifier = PyTorchClassifier(model, nn.CrossEntropyLoss(), (3, 224, 224), 1000, clip_values=(0., 1.))

    # load defense
    fpm = FourierPeakMatrixCollector(PeakMatrixMethod.optimization, ALGO, LIB)
    scaling = ScalingGenerator.create_scaling_approach(x_src.shape, (224, 224, 3), LIB, ALGO)
    defense = PreventionDefenseGenerator.create_prevention_defense(
        defense_type=PreventionTypeDefense.randomfiltering,
        scaler_approach=scaling,
        fourierpeakmatrixcollector=fpm,
        bandwidth=2,
        verbose_flag=False,
        usecythonifavailable=True
    )

    # test raw image
    x_inp = scaling.scale_image(x_src).astype(np.float32).transpose((2, 0, 1)) / 255.
    y_pred = classifier.predict(x_inp[None, ...])
    show('RAW', y_pred, y_img, k=[1, 5])

    # get protected images
    defense.make_image_secure(x_src)
    with ProcessPoolExecutor() as exe:
        output = list(tqdm(exe.map(get, [x_src] * args.repeat), total=args.repeat))

    # test protected images
    x_batch = np.stack(output, axis=0).astype(np.float32).transpose((0, 3, 1, 2)) / 255.
    y_pred = classifier.predict(x_batch)
    show('DEF', y_pred, y_img, k=[1, 5])
