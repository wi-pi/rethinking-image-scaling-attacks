import numpy as np
import torch
import torchvision.transforms.functional as F
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from scaling.ScalingApproach import ScalingApproach
from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries
from torch.autograd import Variable

from scaleadv.bypass.random import resize_to_224x
from scaleadv.datasets.imagenet import create_dataset
import numpy as np
from PIL import Image
from attack.QuadrScaleAttack import QuadraticScaleAttack
from attack.adaptive_attack.AdaptiveAttackPreventionGenerator import AdaptiveAttackPreventionGenerator
from defenses.detection.fourier.FourierPeakMatrixCollector import FourierPeakMatrixCollector, PeakMatrixMethod
from defenses.prevention.PreventionDefenseGenerator import PreventionDefenseGenerator
from defenses.prevention.PreventionDefenseType import PreventionTypeDefense
from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries

from scaleadv.datasets.imagenet import create_dataset
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from itertools import product

import cvxpy as cp
import numpy as np
import numpy.linalg as LA
import torch.nn as nn
from PIL import Image
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier
from attack.area_attack.AreaNormEnumType import AreaNormEnumType
from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries
from torchvision.models import resnet50
from tqdm import tqdm, trange

from scaleadv.bypass.random import resize_to_224x
from scaleadv.datasets.imagenet import IMAGENET_MEAN, IMAGENET_STD
from scaleadv.datasets.imagenet import create_dataset
from scaleadv.models.layers import NormalizationLayer

class ScalingNet(nn.Module):

    def __init__(self, cl: np.ndarray, cr: np.ndarray):
        super(ScalingNet, self).__init__()
        self.cl = nn.Parameter(torch.as_tensor(cl.copy(), dtype=torch.float32), requires_grad=False)
        self.cr = nn.Parameter(torch.as_tensor(cr.copy(), dtype=torch.float32), requires_grad=False)

    def forward(self, inp: torch.Tensor):
        return self.cl @ inp @ self.cr


if __name__ == '__main__':
    TAG = 'SGD'
    RUN_CVX = False
    RUN_SGD = True
    # load data
    dataset = create_dataset(transform=None)
    _, src, _ = dataset[5000]
    _, tgt, _ = dataset[1000]
    src = resize_to_224x(src, more=2)
    x_src = np.array(src)
    x_tgt = np.array(tgt)

    # load scaling & scaled target image
    lib = SuppScalingLibraries.CV
    algo = SuppScalingAlgorithms.LINEAR
    scaling = ScalingGenerator.create_scaling_approach(x_src.shape, (224, 224, 4), lib, algo)
    x_tgt = scaling.scale_image(x_tgt)

    """The following snippet implement scaling-attack with cvxpy
    """
    if RUN_CVX:
        # load scaling attack
        fpm = FourierPeakMatrixCollector(PeakMatrixMethod.optimization, algo, lib)
        scl_attack = QuadraticScaleAttack(eps=1, verbose=False)
        x_scl_cvx, _, _ = scl_attack.attack(x_src, x_tgt, scaling)
        x_scl_cvx_inp = scaling.scale_image(x_scl_cvx)

        # save cvx results
        Image.fromarray(x_scl_cvx).save(f'{TAG}.cvx-attack.png')
        Image.fromarray(x_scl_cvx_inp).save(f'{TAG}.cvx-attack-inp.png')

    """The following snippet implements scaling-attack with torch
    """
    # load network
    model = ScalingNet(scaling.cl_matrix, scaling.cr_matrix).eval().cuda()
    diff = nn.MSELoss()
    src, tgt = map(lambda x: F.to_tensor(x).cuda(), [x_src, x_tgt])

    # attack
    if RUN_SGD:
        att_proxy = Variable(torch.zeros_like(src), requires_grad=True)
        optimizer = torch.optim.Adam([att_proxy], lr=0.01)
        with trange(2000) as pbar:
            for _ in pbar:
                att = (att_proxy.tanh() + 1) * 0.5
                out = model(att)
                loss1 = diff(src, att)
                loss2 = diff(tgt, out)
                loss = loss1 + loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'SRC': f'{loss1.cpu().item():.3f}', 'OUT': f'{loss2.cpu().item():.3f}'})

        # save torch results
        att = np.array(att.detach().cpu() * 255).round().astype(np.uint8).transpose((1, 2, 0))
        att_inp = scaling.scale_image(att)
        Image.fromarray(att).save(f'{TAG}.sgd-attack.png')
        Image.fromarray(att_inp).save(f'{TAG}.sgd-attack-inp.png')






