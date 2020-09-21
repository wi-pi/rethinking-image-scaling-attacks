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

MODEL_PATH = {
    np.inf: 'static/models/imagenet_linf_4.pt',
    2: 'static/models/imagenet_l2_3_0.pt',
}

if __name__ == '__main__':
    # load params
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--id', type=int, required=True, help='ID of the test example.')
    p.add_argument('-t', '--tag', type=str, required=True, help='TAG of this run.')
    args = p.parse_args()

    # attack params
    norm = 2
    pgd_step = 20
    eps_list = list(range(33))
    sigma_list = [e * 2.5 / pgd_step for e in eps_list]
    up_factors = [1.0, 1.5, 2.0, 2.5, 3.0][1:]

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
    dataset = create_dataset(transform=None)

    # batch attack
    total_data = pickle.load(open(f'vary_{args.tag}.pkl', 'rb'))
    for up in up_factors:
        for eps, sigma in zip(eps_list, sigma_list):
            print(f'EPS = {eps}, UP = {up}')
            path = os.path.join('static/results/vary-pgd', f'{args.tag}/{up}/{eps}')
            if norm == np.inf:
                eps, sigma = eps / 255., sigma / 255.

            # process attack
            adv_attack = PGD(classifier, norm, eps + 1e-8, sigma + 1e-8, max_iter=pgd_step)
            saa = ScaleAdvAttack(classifier, adv_attack, lib, algo, input_shape=(224, 224, 3), up_factor=up, save=path)
            try:
                stats = saa.generate_one(dataset, args.id)
            except Exception as e:
                print(e)
                stats = {}

            # store data
            total_data[(eps, up)] = stats
            show_table(stats)
            pickle.dump(total_data, open(f'vary_{args.tag}.pkl', 'wb'))
