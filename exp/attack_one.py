import argparse

import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier
from loguru import logger

from exp.utils import savefig
from scaleadv.datasets import get_imagenet
from scaleadv.datasets.transforms import Align
from scaleadv.models import ScalingLayer
from scaleadv.models.resnet import IMAGENET_MODEL_PATH, resnet50
from scaleadv.scaling import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _ = parser.add_argument
    # Inputs
    _('--id', type=int, required=True, help='ID of the test image.')
    _('--model', default=None, type=str, choices=IMAGENET_MODEL_PATH.keys(), help='Optional robust model.')
    # Scaling
    _('--lib', default='cv', type=str, choices=str_to_lib.keys(), help='Scaling library.')
    _('--alg', default='linear', type=str, choices=str_to_alg.keys(), help='Scaling algorithm.')
    _('--ratio', default=None, type=int, help='Optional fixed scaling ratio.')
    # Standard evasion attack args
    _('--eps', default=20, type=float, help='Maximum l2 perturbation.')
    _('--step', default=30, type=int, help='Maximum steps of the PGD attack.')
    # Scaling attack args
    pass
    # Persistent
    pass
    args = parser.parse_args()

    # Load data
    transform = T.Compose([
        Align(256, args.ratio),
        T.ToTensor(),
        lambda x: np.array(x)[None],  # make a batch
    ])
    dataset = get_imagenet('val', transform)
    x_large, y_large = dataset[args.id]
    logger.info(f'Load source image: id {args.id}, label {y_large}, shape {x_large.shape}, dtype {x_large.dtype}.')

    # Load scaling api
    shape_large = x_large.shape[-2:]
    shape_small = (256, 256)
    api = ScalingAPI(src_shape=x_large.shape[-2:], tgt_shape=(256, 256), lib=args.lib, alg=args.alg)
    # x_small = api(x_large[0])[None]

    # Load network
    scaling_layer = ScalingLayer.from_api(api)
    backbone_network = resnet50(args.model, normalize=True)
    model = nn.Sequential(scaling_layer, backbone_network)
    classifier = PyTorchClassifier(model, nn.CrossEntropyLoss(), x_large.shape[1:], 1000, clip_values=(0, 1))

    # Load attack
    attack = ProjectedGradientDescentPyTorch(
        classifier,
        norm=2,
        eps=args.eps,
        eps_step=args.eps * 2.5 / args.step,
        max_iter=args.step,
        targeted=False,
        verbose=False,
    )

    # Run attack
    adv_large = attack.generate(x_large, np.eye(1000)[[y_large]])
    adv_small = api(adv_large[0])
    savefig(adv_large, 'test1.png')
    savefig(adv_small, 'test2.png')
