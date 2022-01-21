import os
import pickle
from argparse import ArgumentParser

import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from art.estimators.classification import PyTorchClassifier
from loguru import logger

from scaleadv.attacks.sign_opt_new import SignOPT
from scaleadv.datasets import get_imagenet
from scaleadv.datasets.transforms import Align
from scaleadv.defenses import POOLING_MAPS
from scaleadv.models import resnet50, ScalingLayer
from scaleadv.models.resnet import IMAGENET_MODEL_PATH
from scaleadv.scaling import ScalingAPI, str_to_alg, str_to_lib

if __name__ == '__main__':
    p = ArgumentParser()
    _ = p.add_argument
    # Input args
    _('--id', default=-1, type=int, help='set a particular id')
    _('--model', default='none', type=str, choices=IMAGENET_MODEL_PATH.keys(), help='use robust model, optional')
    _('-l', type=int)
    _('-r', type=int)
    _('-s', type=int, default=1)
    _('-g', type=int, default=0)
    # Scaling args
    _('--lib', default='cv', type=str, choices=str_to_lib.keys(), help='scaling libraries')
    _('--alg', default='linear', type=str, choices=str_to_alg.keys(), help='scaling algorithms')
    _('--scale', default=None, type=int, help='set a fixed scale ratio, unset to use the original size')
    # Scaling attack args
    _('--defense', default='none', type=str, choices=POOLING_MAPS.keys(), help='type of defense')
    _('--query', default=25000, type=int, help='query limit')
    _('--tag', default='test', type=str)
    _('--no-smart-noise', action='store_true')
    _('--no-smart-median', action='store_true')
    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.g}'

    # Check test mode
    INSTANCE_TEST = args.id != -1

    # Load data
    transform = T.Compose([Align(224, args.scale), T.ToTensor(), lambda x: np.array(x)[None, ...]])
    dataset = get_imagenet('val' if INSTANCE_TEST else f'val_3', transform)
    id_list = pickle.load(open(f'static/meta/valid_ids.model_{args.model}.scale_3.pkl', 'rb'))[::4]

    # Load test sample
    src, y_src = dataset[id_list[args.id]]
    logger.info(f'Loading source image: id {id_list[args.id]}, label {y_src}, shape {src.shape}, dtype {src.dtype}.')

    # Load network
    class_network = resnet50(robust=args.model, normalize=True).eval().cuda()

    if args.scale != 1:
        # Load scaling
        scaling_api = ScalingAPI(src.shape[-2:], (224, 224), args.lib, args.alg)
        scaling_layer = ScalingLayer.from_api(scaling_api).eval().cuda()

        # Load pooling
        cls = POOLING_MAPS[args.defense]
        pooling_layer = cls.auto(round(scaling_api.ratio) * 2 - 1, scaling_api.mask).eval().cuda()

        # Load network
        big_class_network = nn.Sequential(scaling_layer, class_network).eval().cuda()
        def_class_network = nn.Sequential(pooling_layer, big_class_network).eval().cuda()
    else:
        big_class_network = class_network

    # Test on BIG NO-DEFENSE network.
    classifier = PyTorchClassifier(big_class_network, nn.CrossEntropyLoss(), src.shape[1:], 1000, clip_values=(0, 1))

    # Load attack
    attack = SignOPT(classifier, k=200)
    results = attack.generate(src, y_src, alpha=0.2, beta=0.001, iterations=1000, query_limit=20000)
    x_adv, x_adv_dist, _, nb_queries, x_adv_direction = results
