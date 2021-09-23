import argparse

import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier
from loguru import logger

from exp.utils import savefig
from scaleadv.attacks.carlini import CarliniL2Method
from scaleadv.datasets import get_imagenet
from scaleadv.datasets.transforms import Align
from scaleadv.defenses.preprocessor import MedianFilteringPyTorch
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
    x_small = api(x_large[0])[None]

    # Load network
    scaling_layer = ScalingLayer.from_api(api)
    backbone_network = resnet50(args.model, normalize=True)
    model = nn.Sequential(scaling_layer, backbone_network) if api.ratio != 1 else backbone_network
    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=x_large.shape[1:],
        nb_classes=1000,
        clip_values=(0, 1),
        preprocessing_defences=MedianFilteringPyTorch(api),
    )

    # Load attack
    # attack = ProjectedGradientDescentPyTorch(
    #     classifier,
    #     norm=2,
    #     eps=args.eps,
    #     eps_step=args.eps * 2.5 / args.step,
    #     max_iter=args.step,
    #     targeted=False,
    #     verbose=False,
    # )
    attack = CarliniL2Method(
        classifier,
        confidence=0,
        targeted=False,
        learning_rate=1e-2,
        binary_search_steps=9,
        max_iter=100,#00,
        initial_const=1e-3,
        batch_size=1,
        verbose=False
    )

    # Run attack
    adv_large = attack.generate(x_large, np.eye(1000)[[y_large]])
    adv_small = api(adv_large[0])
    print(classifier.predict(adv_large).argmax(1))

    import scipy.linalg as la
    print('-- large', la.norm(adv_large - x_large))
    print('-- small', la.norm(adv_small - x_small))


    savefig(adv_large, f'test1.{api.ratio}-med.png')
    savefig(adv_small, f'test2.{api.ratio}-med.png')
