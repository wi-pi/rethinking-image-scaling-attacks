import os

import torch.nn as nn
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier
from defenses.prevention.PreventionDefenseType import PreventionTypeDefense
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries
from torchvision.models import resnet50

from scaleadv.attacks.scaleadv import ScaleAdvAttack, show_table
from scaleadv.datasets.imagenet import IMAGENET_MEAN, IMAGENET_STD, create_dataset
from scaleadv.models.layers import NormalizationLayer

ID = 39011
LIB = SuppScalingLibraries.PIL
ALGO = SuppScalingAlgorithms.NEAREST
NORM = 2
NORM_STR = '2'
STEP = 20
EPSILON = 10
EPSILON_STR = '10'
SIGMA = EPSILON * 2.5 / STEP
OUTPUT = f'static/results/bypass/random/L{NORM_STR}-{EPSILON_STR}'

if __name__ == '__main__':
    # load data
    dataset = create_dataset(transform=None)
    _, x_src, y_src = dataset[ID]

    # load SAA
    model = nn.Sequential(NormalizationLayer(IMAGENET_MEAN, IMAGENET_STD), resnet50(pretrained=True)).eval().cuda()
    classifier = PyTorchClassifier(model, nn.CrossEntropyLoss(), (3, 224, 224), 1000, clip_values=(0., 1.))
    attacker = ProjectedGradientDescentPyTorch(classifier, NORM, EPSILON, SIGMA, max_iter=STEP)
    SAA = ScaleAdvAttack(classifier, attacker, LIB, ALGO, (224, 224, 3), save=None)
    defense = PreventionTypeDefense.randomfiltering

    # params
    p1 = f'{OUTPUT}/A'
    os.makedirs(p1, exist_ok=True)
    p2 = f'{OUTPUT}/B'
    os.makedirs(p2, exist_ok=True)

    # attack 1: adv on src
    SAA.save = p1
    stats, x_src_def = SAA.generate_one(dataset, ID, large_inp=None, defense_type=defense, get_defense=100)
    show_table(stats)

    # attack 2: adv on src_def's small
    SAA.save = p2
    stats = SAA.generate_one(dataset, ID, large_inp=x_src_def, defense_type=defense)
    show_table(stats)
