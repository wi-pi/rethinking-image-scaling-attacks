import argparse
import os
import pickle

import numpy as np
import torch.nn as nn
from art.attacks.evasion import ProjectedGradientDescentPyTorch as PGD
from art.estimators.classification import PyTorchClassifier
from prettytable import PrettyTable
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries
from torchvision.models import resnet50

from scaleadv.attacks.scaleadv import ScaleAdvAttack
from scaleadv.datasets.imagenet import IMAGENET_MEAN, IMAGENET_STD, create_dataset
from scaleadv.models.layers import NormalizationLayer
from scaleadv.tests.gen_adv_pgd import get_model

OUTPUT_PATH = 'static/results/saa_pgd/'
DATASET_PATH = 'static/datasets/imagenet/val/'
MODEL_PATH = {
    np.inf: 'static/models/imagenet_linf_4.pt',
    2: 'static/models/imagenet_l2_3_0.pt',
}

if __name__ == '__main__':
    # load params
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--norm', type=int, default=np.inf, help='Norm of PGD attack.')
    p.add_argument('-e', '--eps', type=float, required=True, help='Epsilon.')
    p.add_argument('-s', '--sigma', type=float, required=True, help='Sigma.')
    p.add_argument('-i', '--iter', type=int, required=True, help='PGD iterations.')
    p.add_argument('-t', '--tag', type=str, required=True, help='Tag for this run.')
    p.add_argument('--skip', type=int, default=50, help='Test on every SKIP examples.')
    args = p.parse_args()
    print(args)

    # prepare output dir
    path = os.path.join(OUTPUT_PATH, args.tag)
    os.makedirs(path, exist_ok=True)

    # load AA
    model = nn.Sequential(
        NormalizationLayer(IMAGENET_MEAN, IMAGENET_STD),
        get_model(weights_file=MODEL_PATH[args.norm])
    ).eval()
    classifier = PyTorchClassifier(model, nn.CrossEntropyLoss(), (3, 224, 224), 1000, clip_values=(0., 1.))
    attacker = PGD(classifier, args.norm, args.eps, args.sigma, max_iter=args.iter)

    # load SA
    lib = SuppScalingLibraries.PIL
    algo = SuppScalingAlgorithms.NEAREST

    # load SAA
    saa = ScaleAdvAttack(classifier, attacker, lib, algo, input_shape=(224, 224, 3), save=path)

    # test
    dataset = create_dataset(transform=None)
    indices = range(0, len(dataset), args.skip)
    stats = saa.generate(dataset, indices)

    # save stats
    filename = os.path.join(path, 'stats_final.pkl')
    pickle.dump(stats, open(filename, 'wb'))
