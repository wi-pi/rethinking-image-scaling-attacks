import argparse

import numpy as np
import scipy.linalg as la
import torch.nn as nn
import torchvision.transforms as T
from art.defences.preprocessor import SpatialSmoothingPyTorch, JpegCompression
from art.estimators.classification import PyTorchClassifier
from loguru import logger

from exp.utils import savefig
from scaleadv.attacks.carlini import CarliniL2Method
from scaleadv.datasets import get_imagenet
from scaleadv.datasets.transforms import Align
from scaleadv.defenses.preprocessor import MedianFilteringExact, MedianFilteringBPDA
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
    x_large, y = dataset[args.id]
    y_onehot = np.eye(1000)[[y]]
    logger.info(f'Load source image: id {args.id}, label {y}, shape {x_large.shape}, dtype {x_large.dtype}.')

    # Load scaling api
    shape_large = x_large.shape[-2:]
    shape_small = (256, 256)
    api = ScalingAPI(src_shape=x_large.shape[-2:], tgt_shape=(256, 256), lib=args.lib, alg=args.alg)
    x_small = api(x_large[0])[None]

    # Load network
    small_network = resnet50(args.model, normalize=True)
    large_network = nn.Sequential(ScalingLayer.from_api(api), small_network)
    kwargs = dict(
        loss=nn.CrossEntropyLoss(),
        input_shape=x_large.shape[1:],
        nb_classes=1000,
        clip_values=(0, 1),
    )

    # Load LR classifier
    kwargs['model'] = small_network
    classifier_small = PyTorchClassifier(**kwargs)

    # Load HR classifiers
    kwargs['model'] = large_network
    # Not protected
    classifier_large_unprotected = PyTorchClassifier(**kwargs)
    # JPEG protected
    jpeg = JpegCompression(clip_values=(0, 1), quality=30)
    classifier_large_jpeg = PyTorchClassifier(**kwargs, preprocessing_defences=jpeg)

    # Load attacks
    # attack_cls = ProjectedGradientDescentPyTorch
    # kwargs = dict(
    #     norm=2,
    #     eps=args.eps,
    #     eps_step=args.eps * 2.5 / args.step,
    #     max_iter=args.step,
    #     targeted=False,
    #     verbose=False,
    # )
    attack_cls = CarliniL2Method
    kwargs = dict(
        confidence=0,
        max_iter=200,
        binary_search_steps=20,
        targeted=False,
        verbose=False,
    )

    attack_small = attack_cls(classifier_small, **kwargs)
    attack_large_unprotected = attack_cls(classifier_large_unprotected, **kwargs)
    attack_large_jpeg = attack_cls(classifier_large_jpeg, **kwargs)

    # Attack small
    adv_small = attack_small.generate(x_small, y_onehot)
    print('Attack small:', la.norm(adv_small - x_small), classifier_small.predict(adv_small).argmax(1))
    savefig(adv_small, 'test0.png')

    classifier_list = [
        classifier_large_unprotected,
        classifier_large_jpeg,
    ]
    attack_list = [
        attack_large_unprotected,
        attack_large_jpeg,
    ]
    for i, (a, c) in enumerate(zip(attack_list, classifier_list), start=1):
        adv_large = a.generate(x_large, y_onehot)
        savefig(adv_large, f'test{i}.png')
        print(f'Attack {i}:', la.norm(adv_large - x_large) / api.ratio,
              [c.predict(adv_large).argmax(1) for c in classifier_list])
