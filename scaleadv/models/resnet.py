from typing import Optional

import torch
import torchvision.models as models

IMAGENET_MODEL_PATH = {
    'inf': 'static/models/imagenet_linf_4.pt',
    '2': 'static/models/imagenet_l2_3_0.pt',
    'smooth': 'static/models/models/imagenet/resnet50/noise_1.00/checkpoint.pth.tar',
    'smooth-adv': 'static/models/pretrained_models/imagenet/PGD_1step/imagenet/eps_255/resnet50/noise_1.00/checkpoint.pth.tar'
}


def resnet50_imagenet(robust: Optional[str] = None):
    pretrained = robust is None
    network = models.resnet50(pretrained=pretrained)

    if robust is not None:
        assert robust in IMAGENET_MODEL_PATH
        if robust in ['2', 'inf']:
            prefix = 'module.model.'
            key = 'model'
        elif robust in ['smooth', 'smooth-adv']:
            prefix = '1.module.'
            key = 'state_dict'
        else:
            raise NotImplementedError
        ckpt = torch.load(IMAGENET_MODEL_PATH[robust]).get(key, {})
        ckpt = {k.replace(prefix, ''): v for k, v in ckpt.items() if k.startswith(prefix)}
        network.load_state_dict(ckpt)

    return network
