"""
This script shows how to bypass the Median-Filter defense.
1. Denote by `x_src` the source image.
2. Applying the defense on `x_src`, and gets `x_src_def`.
3. Down-scale `x_src_def` to the expected size, and gets `x_src_def_small`.
   Now, this down-scaled image should contain exactly the median value of each window.
4. Apply PGD attack on `x_src_def_small`, and gets `x_src_def_small_adv`.
5. Apply adaptive-attack with source `x_src` and target `x_src_def_small_adv`.
6. Return the final attack image `x_att`.

Notes.
1. We only consider non-overlapping windows here.

# attack on src
+--------+--------+--------+--------+--------+--------+---------+--------+
|  100   | y_pred |  Linf  |   L2   |  PSNR  |  SSIM  | MS-SSIM | LPIPS  |
+--------+--------+--------+--------+--------+--------+---------+--------+
| x_inp  |  100   |  0.000 |  0.000 | 80.000 |  1.000 |   1.000 |  0.000 |
| x_adv  |   99   |  0.031 |  6.855 | 35.056 |  0.961 |   0.994 |  0.031 |
| x_src  |  100   |  0.000 |  0.000 | 80.000 |  1.000 |   1.000 |  0.000 |
| x_src* |  100   |  0.729 | 28.424 | 29.677 |  0.960 |   0.993 |  0.027 |
| x_scl  |   99   |  0.024 |  4.517 | 45.651 |  0.992 |   0.999 |  0.002 |
| x_scl* |  100   |  0.729 | 28.424 | 29.677 |  0.960 |   0.993 |  0.027 |
| x_ada  |   99   |  0.745 | 40.208 | 26.664 |  0.920 |   0.981 |  0.058 |
| x_ada* |   99   |  0.745 | 41.436 | 26.403 |  0.916 |   0.980 |  0.058 |
+--------+--------+--------+--------+--------+--------+---------+--------+
# attack on src_def
+--------+--------+--------+--------+--------+--------+---------+--------+
|  100   | y_pred |  Linf  |   L2   |  PSNR  |  SSIM  | MS-SSIM | LPIPS  |
+--------+--------+--------+--------+--------+--------+---------+--------+
| x_inp  |  100   |  0.000 |  0.000 | 80.000 |  1.000 |   1.000 |  0.000 |
| x_adv  |   99   |  0.745 | 29.155 | 22.482 |  0.775 |   0.958 |  0.133 |
| x_src  |  100   |  0.000 |  0.000 | 80.000 |  1.000 |   1.000 |  0.000 |
| x_src* |  100   |  0.729 | 28.424 | 29.677 |  0.960 |   0.993 |  0.027 |
| x_scl  |   99   |  0.737 | 27.533 | 29.954 |  0.959 |   0.993 |  0.026 |
| x_scl* |  100   |  0.729 | 28.424 | 29.677 |  0.960 |   0.993 |  0.027 |
| x_ada  |   99   |  0.737 | 28.304 | 29.714 |  0.946 |   0.991 |  0.036 |
| x_ada* |   99   |  0.729 | 28.463 | 29.665 |  0.946 |   0.991 |  0.035 |
+--------+--------+--------+--------+--------+--------+---------+--------+
"""
import os

import numpy as np
import torch.nn as nn
from PIL import Image
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
EPSILON = 20
EPSILON_STR = '20'
SIGMA = EPSILON * 2.5 / STEP
OUTPUT = f'static/results/bypass/median/L{NORM_STR}-{EPSILON_STR}'

if __name__ == '__main__':
    # load data
    dataset = create_dataset(transform=None)
    _, x_src, y_src = dataset[ID]

    # load SAA
    model = nn.Sequential(NormalizationLayer(IMAGENET_MEAN, IMAGENET_STD), resnet50(pretrained=True)).eval().cuda()
    classifier = PyTorchClassifier(model, nn.CrossEntropyLoss(), (3, 224, 224), 1000, clip_values=(0., 1.))
    attacker = ProjectedGradientDescentPyTorch(classifier, NORM, EPSILON, SIGMA, max_iter=STEP)
    SAA = ScaleAdvAttack(classifier, attacker, LIB, ALGO, (224, 224, 3), save=None)
    defense = PreventionTypeDefense.medianfiltering

    # params
    p1 = f'{OUTPUT}/A'
    os.makedirs(p1, exist_ok=True)
    p2 = f'{OUTPUT}/B'
    os.makedirs(p2, exist_ok=True)

    # attack 1: adv on src
    SAA.save = p1
    stats = SAA.generate_one(dataset, ID, large_inp=None, defense_type=defense)
    show_table(stats)

    # load src_def
    x_src_def = np.array(Image.open(f'{p1}/{ID}.x_src_def.png'))

    # attack 2: adv on src_def's small
    SAA.save = p2
    stats = SAA.generate_one(dataset, ID, large_inp=x_src_def, defense_type=defense)
    show_table(stats)
