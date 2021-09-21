import argparse

import numpy as np
import torchvision.transforms as T
from loguru import logger

from scaleadv.datasets import get_imagenet
from scaleadv.datasets.transforms import Align
from scaleadv.models.resnet import IMAGENET_MODEL_PATH
from scaleadv.scaling import ScalingAPI
from scaleadv.scaling.enum import str_to_alg, str_to_lib

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
    api = ScalingAPI(x_large.shape[-2:], (256, 256), args.lib, args.alg)
    x_small = api(x_large[0])[None]

    print(x_large.shape)
    print(x_small.shape)
