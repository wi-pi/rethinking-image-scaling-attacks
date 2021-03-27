from argparse import ArgumentParser
import torch

import numpy as np
import torchvision.transforms as T
from art.attacks.evasion import ProjectedGradientDescentPyTorch as PGD
from art.estimators.classification import PyTorchClassifier
from loguru import logger
from torch.nn import DataParallel

from scaleadv.attacks.core import ScaleAttack
from scaleadv.attacks.sign_opt import OPT_attack_sign_SGD_v2
from scaleadv.datasets import get_imagenet
from scaleadv.datasets.transforms import Align
from scaleadv.defenses import POOLING_MAPS
from scaleadv.evaluate import Evaluator
from scaleadv.models import resnet50, ScalingLayer
from scaleadv.models.resnet import IMAGENET_MODEL_PATH
from scaleadv.scaling import ScalingAPI, ScalingLib, ScalingAlg
import torchvision.transforms.functional as F
import torch.nn as nn

class ModelAdaptor(object):

    def __init__(self, model: PyTorchClassifier):
        self.model = model

    def predict_label(self, x: torch.Tensor):
        x = x.clip(0, 1)
        p = self.model.predict(x.cpu().numpy()).argmax(1)
        return torch.tensor(p).cuda()

if __name__ == '__main__':
    p = ArgumentParser()
    _ = p.add_argument
    # Input args
    _('--id', type=int, required=True, help='ID of test image')
    # Scaling args
    _('--lib', default='cv', type=str, choices=ScalingLib.names(), help='scaling libraries')
    _('--alg', default='linear', type=str, choices=ScalingAlg.names(), help='scaling algorithms')
    _('--scale', default=None, type=int, help='set a fixed scale ratio, unset to use the original size')
    args = p.parse_args()

    # Load data
    transform = T.Compose([Align(224, args.scale), T.ToTensor(), lambda x: np.array(x)[None, ...]])
    dataset = get_imagenet('val', transform)
    src, y_src = dataset[args.id]
    logger.info(f'Loading source image: id {args.id}, label {y_src}, shape {src.shape}, dtype {src.dtype}.')

    # Load scaling
    scaling_api = ScalingAPI(src.shape[-2:], (224, 224), args.lib, args.alg)
    scaling_layer = ScalingLayer.from_api(scaling_api)

    # Load network
    class_network = resnet50(robust=None, normalize=True)
    class_network = nn.Sequential(scaling_layer, class_network).eval().cuda()

    # Prepare adv attack
    classifier = PyTorchClassifier(class_network, nn.CrossEntropyLoss(), src.shape[1:], 1000, clip_values=(0, 1))

    # Black-box Attack
    attacker = OPT_attack_sign_SGD_v2(model=ModelAdaptor(classifier))

    x0 = torch.tensor(src).cuda()
    y0 = torch.tensor([y_src]).cuda()
    x_adv, _ = attacker.attack_untargeted(x0, y0)

    # post-process
    x_adv = x_adv.cpu().clip(0, 1)
    F.to_pil_image(x_adv[0]).save('bb.png')
    print('src:', y_src)
    print('adv:', classifier.predict(x_adv).argmax(1)[0])
    from IPython import embed; embed(using=False); exit()
