from typing import Optional

import torch
import torchvision.models as models

IMAGENET_MODEL_PATH = {
    'inf': 'static/models/imagenet_linf_4.pt',
    '2': 'static/models/imagenet_l2_3_0.pt',
}


def resnet50_imagenet(robust: Optional[str] = None):
    pretrained = robust is None
    network = models.resnet50(pretrained=pretrained)

    if robust is not None:
        assert robust in IMAGENET_MODEL_PATH
        prefix = 'module.model.'
        ckpt = torch.load(IMAGENET_MODEL_PATH[robust]).get('model', {})
        ckpt = {k.replace(prefix, ''): v for k, v in ckpt.items() if k.startswith(prefix)}
        network.load_state_dict(ckpt)

    return network
