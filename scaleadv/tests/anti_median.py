import numpy as np
import torch.nn as nn
from PIL import Image
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries
from torchvision.models import resnet50

from scaleadv.attacks.scaleadv import ScaleAdvAttack, show_table
from scaleadv.datasets.imagenet import create_dataset, IMAGENET_MEAN, IMAGENET_STD
from scaleadv.models.layers import NormalizationLayer

if __name__ == '__main__':
    # load dataset
    dataset = create_dataset(transform=None)
    _, x_img, y_true = dataset[5000]

    # load AA
    model = nn.Sequential(
        NormalizationLayer(IMAGENET_MEAN, IMAGENET_STD),
        resnet50(pretrained=True)
    ).eval()
    classifier = PyTorchClassifier(model, nn.CrossEntropyLoss(), (3, 224, 224), 1000, clip_values=(0., 1.))
    #attacker = ProjectedGradientDescentPyTorch(classifier, np.inf, 16 / 255., 2 / 255., max_iter=20)
    attacker = ProjectedGradientDescentPyTorch(classifier, 2, 6., 0.75, max_iter=20)

    # load SA
    lib = SuppScalingLibraries.PIL
    algo = SuppScalingAlgorithms.NEAREST

    # load SAA
    path_a = 'static/results/anti-median/A/L2-6'
    saa = ScaleAdvAttack(classifier, attacker, lib, algo, input_shape=(224, 224, 3), save=path_a)

    # attack 1: adv on src
    stats = saa.generate_one(dataset, 5000)
    show_table(stats)

    # load SAA
    path_b = 'static/results/anti-median/B/L2-6'
    saa = ScaleAdvAttack(classifier, attacker, lib, algo, input_shape=(224, 224, 3), save=path_b)

    # attack 2: adv on src_def_down
    x_src_def = Image.open(f'{path_a}/5000.x_src_def.png')
    x_src_def = np.array(x_src_def)
    stats = saa.generate_one(dataset, 5000, large_inp=x_src_def)
    show_table(stats)
