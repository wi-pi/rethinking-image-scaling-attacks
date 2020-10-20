from RandomizedSmoothing.utils.regularizers import get_tv, get_color, get_sim
from torch.nn import DataParallel

from scaleadv.bypass.random import resize_to_224x
import torchvision.transforms.functional as F
from scaleadv.datasets.imagenet import create_dataset
from scaleadv.experiments.scaling_attack_sgd import ScalingNet
from scaleadv.models.layers import RandomPool2d
import numpy as np
import numpy.linalg as LA
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier
from attack.QuadrScaleAttack import QuadraticScaleAttack
from defenses.detection.fourier.FourierPeakMatrixCollector import FourierPeakMatrixCollector, PeakMatrixMethod
from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries
from torch.autograd import Variable
from torchvision.models import resnet50
from tqdm import trange

from scaleadv.bypass.random import resize_to_224x
from scaleadv.datasets.imagenet import create_dataset, IMAGENET_MEAN, IMAGENET_STD
from scaleadv.models.layers import MedianPool2d, RandomPool2d
from scaleadv.models.layers import NormalizationLayer
from scaleadv.tests.gen_adv_pgd import get_model

MODEL_PATH = {
    np.inf: 'static/models/imagenet_linf_4.pt',
    2: 'static/models/imagenet_l2_3_0.pt',
}

def test(x: torch.Tensor, n:int):
    with torch.no_grad():
        x = torch.clamp(x, 0, 1)
        xs = pooling(x.cpu().repeat(n,1,1,1)).cuda()
        pred = model(xs).argmax(1)
    return pred.cpu().numpy()

def attack(x: torch.Tensor, y: int, target: int, fix_pooling: torch.Tensor =None, desc: str='Attack'):
    assert len(x.shape) == 3
    if fix_pooling is not None:
        assert len(fix_pooling.shape) == 4
    # prepare optim
    delta = torch.rand_like(x) - 0.5
    delta.requires_grad_()
    optimizer = torch.optim.Adam([delta], lr=0.1)
    y_tgt = torch.LongTensor([target]).repeat((N, )).to(device=x.device)

    # attack iters
    with trange(T, desc=desc) as pbar:
        for _ in pbar:
            # generate current input
            if fix_pooling is None:
                att = x + delta
                att = pooling(att.cpu().repeat(N,1,1,1)).cuda()
            else:
                att = fix_pooling + delta
            # forward
            pred = model(att)
            # compute loss
            loss_cl = nn.functional.cross_entropy(pred, y_tgt, reduction='mean').mean()
            loss_tv, loss_color, loss_sim = [f(delta) for f in [get_tv, get_color, get_sim]]
            loss = loss_cl + 0.1 * loss_tv + 20.0 * loss_color + 10.0 * loss_sim
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # logging
            pred = pred.argmax(1).cpu().numpy()
            pbar.set_postfix({
                'CLS': f'{loss_cl.cpu().item():.3f}',
                'TV': f'{loss_tv.cpu().item():.3f}',
                'COLOR': f'{loss_color.cpu().item():.3f}',
                'SIM': f'{loss_sim.cpu().item():.3f}',
                f'PRED-{y}': f'{(pred == y).mean():.2%}',
                f'PRED-{target}': f'{(pred == target).mean():.2%}',
            })

    return torch.clamp(x + delta, 0, 1).detach()


def iterative_attack(x: torch.Tensor, y:int, target:int):
    att = x.clone().detach()
    for i in range(5):
        att_batch = pooling(att.cpu().repeat(N,1,1,1)).cuda()
        att = attack(att, y, target, fix_pooling=att_batch, desc=f'Attack-{i}')
        # test
        pred = test(att, n=N)
        print(f'Test {y}: {np.mean(pred == y):.2%}')
        print(f'Test {target}: {np.mean(pred == target):.2%}')
    return att





if __name__ == '__main__':
    # params
    N = 256      # sample numbers
    T = 300      # inner attack iters
    EPS = 1e-6   # tanh
    ID = 5000    # src image
    TGT = 200    # tgt class
    TAG = f'ADA-RANDOM.{ID}'

    # load data & align to 224
    dataset = create_dataset(transform=None)
    _, src, y_src = dataset[ID]
    src = resize_to_224x(src, more=1)
    src = np.array(src)

    # load scaling tools
    lib = SuppScalingLibraries.CV
    algo = SuppScalingAlgorithms.LINEAR
    scaling = ScalingGenerator.create_scaling_approach(src.shape, (224, 224, 4), lib, algo)

    # get modified pixel's mask
    cl, cr = scaling.cl_matrix, scaling.cr_matrix
    cli, cri = map(LA.pinv, [cl, cr])
    mask = np.round(cli @ np.ones((224, 224)) @ cri).astype(np.uint8)

    # load models
    pooling = MedianPool2d(7, 1, 3, mask=mask)
    pooling = RandomPool2d(7, 1, 3, mask=mask)
    # pooling = lambda x: x
    model = nn.Sequential(
        ScalingNet(scaling.cl_matrix, scaling.cr_matrix),
        NormalizationLayer(IMAGENET_MEAN, IMAGENET_STD),
        get_model(weights_file=MODEL_PATH[2]),
        # resnet50(pretrained=True)
    ).eval()
    model = DataParallel(model).cuda()

    # N = 1
    """Direct Attack"""
    #att = attack(F.to_tensor(src).cuda(), y_src, TGT)
    # F.to_pil_image(att.cpu()).save('att-med-res.png')
    # exit()

    """Iterative Attack"""
    att = iterative_attack(F.to_tensor(src).cuda(), y_src, TGT)
    F.to_pil_image(att.cpu()).save('att-iter-ran-rob.png')
    exit()
