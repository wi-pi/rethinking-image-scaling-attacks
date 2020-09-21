import argparse
import os
import pickle

import numpy as np
import torch.nn as nn
from art.attacks.evasion import ProjectedGradientDescentPyTorch as PGD
from art.estimators.classification import PyTorchClassifier
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries

from scaleadv.attacks.scaleadv import ScaleAdvAttack, show_table
from scaleadv.datasets.imagenet import IMAGENET_MEAN, IMAGENET_STD, create_dataset
from scaleadv.models.layers import NormalizationLayer
from scaleadv.tests.gen_adv_pgd import get_model

DATASET_PATH = 'static/datasets/imagenet-600/'
OUTPUT_PATH = 'static/results/vary-pgd-600/'
MODEL_PATH = {
    np.inf: 'static/models/imagenet_linf_4.pt',
    2: 'static/models/imagenet_l2_3_0.pt',
}

if __name__ == '__main__':
    # load params
    p = argparse.ArgumentParser()
    p.add_argument('-u', '--up', type=int, required=True, help='Up factor to test in {2,3,4,5,7}.')
    args = p.parse_args()

    # attack params
    norm = np.inf  # NOTE: we only consider Linf attack right now
    pgd_step = 20
    eps_list = list(range(33))
    sigma_list = [e * 2.5 / pgd_step for e in eps_list]

    # load classifier
    model = nn.Sequential(
        NormalizationLayer(IMAGENET_MEAN, IMAGENET_STD),
        get_model(weights_file=MODEL_PATH[norm])
    ).eval()
    classifier = PyTorchClassifier(model, nn.CrossEntropyLoss(), (3, 224, 224), 1000, clip_values=(0., 1.))

    # load SA
    lib = SuppScalingLibraries.PIL
    algo = SuppScalingAlgorithms.NEAREST

    # load data
    dataset = create_dataset(root=f'{DATASET_PATH}/{args.up}', transform=None)

    # for the given ratio data, run every eps for every images
    DUMP = f'{OUTPUT_PATH}/{args.up}.pkl'
    for eps, sigma in zip(eps_list, sigma_list):
        print(f'UP = {args.up}, EPS = {eps}')
        path = f'{OUTPUT_PATH}/{args.up}'
        eps, sigma = eps / 255., sigma / 255.

        # process attack
        adv_attack = PGD(classifier, norm, eps + 1e-8, sigma + 1e-8, max_iter=pgd_step)
        saa = ScaleAdvAttack(classifier, adv_attack, lib, algo, input_shape=(224, 224, 3), save=path)
        stats = saa.generate(dataset, range(len(dataset)))
