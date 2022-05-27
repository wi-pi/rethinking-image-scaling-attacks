import argparse
import os
import pickle

import numpy as np
import scipy.linalg as la
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from art.estimators.classification import PyTorchClassifier
from loguru import logger
from torchvision.transforms.functional import to_tensor

from exp.utils import savefig
from scaleadv.attacks.core import ScaleAttack
from scaleadv.datasets import get_imagenet
from scaleadv.datasets.transforms import Align
from scaleadv.defenses import POOLING_MAPS
from scaleadv.models.resnet import IMAGENET_MODEL_PATH, resnet50
from scaleadv.scaling import *


def attack_one(id, setid=False):
    # Load the original large image & true label.
    x_large, y_src = dataset[id] if setid else dataset[id_list[id]]
    logger.info(f'Loading source image: id {id}, label {y_src}, shape {x_large.shape}, dtype {x_large.dtype}.')

    # Load the pre-generated small adversarial example.
    x_small_adv = to_tensor(Image.open(f'{pref}.png')).numpy()[None]

    # Load scaling api & layer & down-scaled original image.
    scaling_api = ScalingAPI(x_large.shape[-2:], (224, 224), args.lib, args.alg)
    x_small = scaling_api(x_large[0])[None]

    # Load networks
    pooling_layer = POOLING_MAPS['none'].auto(round(scaling_api.ratio) * 2 - 1, scaling_api.mask)

    # Load Scale-Adv
    attack = ScaleAttack(scaling_api, pooling_layer, small_network, verbose=True)

    # Load classifiers
    classifier_small = PyTorchClassifier(small_network, nn.CrossEntropyLoss(), x_small.shape[1:], 1000,
                                         clip_values=(0, 1))
    classifier_large = attack.classifier_big
    logger.info(f'Predict x_small as {classifier_small.predict(x_small).argmax(1)}.')
    logger.info(f'Predict x_small_adv as {classifier_small.predict(x_small_adv).argmax(1)}')
    logger.info(f'  with {la.norm(x_small_adv - x_small)} L2 perturbation.')

    # Run attack
    adv_large = attack.hide(src=x_large, tgt=x_small_adv, src_label=y_src, tgt_label=None,
                            max_iter=10000, lr=0.01, weight=1000.0)
    logger.info(f'Predict x_large_adv as {classifier_large.predict(adv_large).argmax(1)}.')
    logger.info(f'  with {la.norm(adv_large - x_large) / 3} L2 perturbation.')

    savefig(adv_large, f'{pref}_hide_ratio3.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _ = parser.add_argument
    # Inputs
    _('--id', default=-1, type=int, help='ID of the test image.')
    _('--model', default='none', type=str, choices=IMAGENET_MODEL_PATH.keys(), help='Optional robust model.')
    _('-l', type=int)
    _('-r', type=int)
    _('-s', type=int, default=1)
    _('-g', type=int, default=0)
    # Scaling
    _('--lib', default='cv', type=str, choices=str_to_lib.keys(), help='Scaling library.')
    _('--alg', default='linear', type=str, choices=str_to_alg.keys(), help='Scaling algorithm.')
    # Standard evasion attack args
    _('--query', default=25000, type=int, help='query limit')
    # Scaling attack args
    pass
    # Persistent
    pass
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.g}'

    # Check test mode
    INSTANCE_TEST = args.id != -1

    # Load data
    transform = T.Compose([Align(256, 3), T.ToTensor(), lambda x: np.array(x)[None, ...]])
    dataset = get_imagenet('val' if INSTANCE_TEST else 'val_3', transform)
    id_list = pickle.load(open(f'static/meta/valid_ids.model_{args.model}.scale_3.pkl', 'rb'))[::2]

    root = 'static/simba'
    os.makedirs(root, exist_ok=True)

    # Load network
    small_network = resnet50(robust=args.model, normalize=True).eval().cuda()

    # attack one
    if INSTANCE_TEST:
        pref = f'simba_test.{args.id}'
        attack_one(args.id, setid=True)
        exit()

    # attack each one
    for i in range(args.l, args.r, args.s):
        pref = f'{root}/{i}.{args.lib}.{args.alg}.ratio_1'
        attack_one(i, setid=False)
