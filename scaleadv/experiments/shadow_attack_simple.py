"""
This script test shadow attack on plain images.
Two implementations are used: official and art.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from art.attacks.evasion import ShadowAttack as ShadowAttack_ART
from art.estimators.classification import PyTorchClassifier
from torch.nn import DataParallel
from torchvision.models import resnet50

from scaleadv.attacks.shadow import SmoothAttack as ShadowAttack_Official
from scaleadv.datasets.imagenet import create_dataset, IMAGENET_MEAN, IMAGENET_STD
from scaleadv.models.layers import NormalizationLayer


def save(name, t):
    x = np.array(t * 255).round().astype(np.uint8).transpose((1, 2, 0))
    Image.fromarray(x).save(name)


if __name__ == '__main__':
    ID = 5000
    pth = 'static/results/experiments/shadow_simple'
    os.makedirs(pth, exist_ok=True)

    # load data to [0, 255] ndarray
    dataset = create_dataset(transform=T.Resize((224, 224)))
    _, x, y = dataset[ID]
    x = (np.array(x) / 255.).astype(np.float32).transpose((2, 0, 1))[None, ...]
    print('x', x.shape)
    print('y', y)

    # load classifier
    model = nn.Sequential(NormalizationLayer(IMAGENET_MEAN, IMAGENET_STD), resnet50(pretrained=True)).eval()
    model = DataParallel(model).cuda()
    classifier = PyTorchClassifier(model, nn.CrossEntropyLoss(), (3, 224, 224), 1000, clip_values=(0., 1.))

    # load attack
    SIGMA = 0.3
    STEP = 300
    BATCH = 400
    LR = 0.1
    art_attack = ShadowAttack_ART(classifier, sigma=SIGMA, nb_steps=STEP, learning_rate=LR, batch_size=BATCH, targeted=False)
    off_attack = ShadowAttack_Official(model)

    # test benign (need [BCHW] ndarray)
    print('y_pred', classifier.predict(x).argmax(1))
    print('y_pred_noise', end='\t')
    for _ in range(20):
        x_noise = x + np.random.normal(scale=SIGMA, size=x.shape).astype(x.dtype)
        print(classifier.predict(x_noise).argmax(1)[0], end=' ')
    print()

    # test art (need [BCHW] ndarray)
    x_adv = art_attack.generate(x)
    print('y_adv', classifier.predict(x_adv).argmax(1))
    print('y_adv_noise', end='\t')
    for _ in range(20):
        x_adv_noise = x_adv + np.random.normal(scale=SIGMA, size=x_adv.shape).astype(x_adv.dtype)
        print(classifier.predict(x_adv_noise).argmax(1)[0], end=' ')
    print()
    save(f'{pth}/{ID}_adv_art.png', x_adv[0])
    exit()

    # test official
    # NOTE: this is targeted by default, check impl if possible.
    x_inp = torch.as_tensor(x[0], dtype=torch.float32)
    x_adv = off_attack.perturb(x_inp, y, sigma=SIGMA, batch=BATCH, print_stats=True)
    x_adv = np.array(x_adv)[None, ...]
    print('y_adv', classifier.predict(x_adv).argmax(1))
    for _ in range(20):
        x_adv_noise = x_adv + np.random.normal(scale=SIGMA, size=x_adv.shape).astype(x_adv.dtype)
        print('y_adv_noise', classifier.predict(x_adv_noise).argmax(1))
    save(f'{pth}/{ID}_inp.png', x_inp)
    save(f'{pth}/{ID}_adv_off.png', x_adv[0])
