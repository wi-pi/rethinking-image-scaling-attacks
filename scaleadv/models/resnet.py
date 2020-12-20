from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
from loguru import logger

from scaleadv.models.layers import NormalizationLayer

IMAGENET_MODEL_PATH = {
    'none': None,
    '2': 'static/models/imagenet_l2_3_0.pt',
    'inf': 'static/models/imagenet_linf_4.pt',
    'smooth': 'static/models/models/imagenet/resnet50/noise_1.00/checkpoint.pth.tar',
    'smooth-adv': 'static/models/pretrained_models/imagenet/PGD_1step/imagenet/eps_255/resnet50/noise_1.00/checkpoint.pth.tar'
}


def resnet50(robust: Optional[str] = None, normalize: bool = True):
    if robust is not None:
        if robust not in IMAGENET_MODEL_PATH:
            raise ValueError(f'Cannot find robust model "{robust}".')
    weight_file = IMAGENET_MODEL_PATH.get(robust, None)

    if weight_file:
        if robust in ['2', 'inf']:
            prefix = 'module.model.'
            key = 'model'
        elif robust in ['smooth', 'smooth-adv']:
            prefix = '1.module.'
            key = 'state_dict'
        else:
            raise NotImplementedError
        logger.info(f'Loading weight file from "{weight_file}".')
        network = models.resnet50(pretrained=False)
        weight = torch.load(weight_file).get(key, {})
        weight = {k.replace(prefix, ''): v for k, v in weight.items() if k.startswith(prefix)}
        network.load_state_dict(weight)
    else:
        logger.info(f'Loading torchvision pretrained weights.')
        network = models.resnet50(pretrained=True)

    if normalize:
        layer = NormalizationLayer.preset('imagenet')
        network = nn.Sequential(layer, network)

    return network
