from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models
from loguru import logger

from src.models.layers import NormalizationLayer

IMAGENET_MODEL_PATH = {
    'nature': None,
    'robust-2': 'static/models/imagenet_l2_3_0.pt',
    'robust-inf': 'static/models/imagenet_linf_4.pt',
}


def imagenet_resnet50(model_name: str | None = None, normalize: bool = True):
    if model_name is not None:
        if model_name not in IMAGENET_MODEL_PATH:
            raise ValueError(f'Cannot find model "{model_name}".')
    weight_file = IMAGENET_MODEL_PATH.get(model_name, None)

    if weight_file:
        match model_name:
            case 'robust-2' | 'robust-inf':
                prefix, key = 'module.model.', 'model'
            case 'smooth' | 'smooth-adv':
                prefix, key = '1.module.', 'state_dict'
            case _:
                prefix, key = '', ''

        logger.info(f'Loading weight file from "{weight_file}".')
        network = models.resnet50(pretrained=False)
        ckpt = torch.load(weight_file).get(key, {})
        ckpt = OrderedDict({k.removeprefix(prefix): v for k, v in ckpt.items()})
        network.load_state_dict(ckpt)

    else:
        logger.info(f'Loading torchvision pretrained weights.')
        network = models.resnet50(pretrained=True)

    if normalize:
        name = 'imagenet'
        logger.info(f'Loading normalization layer for "{name}".')
        layer = NormalizationLayer.preset(name)
        network = nn.Sequential(layer, network)

    return network
