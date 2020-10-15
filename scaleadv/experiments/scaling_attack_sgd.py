import torch
import numpy as np
import numpy.linalg as LA
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image
from RandomizedSmoothing.utils.regularizers import get_color, get_sim, get_tv
from attack.QuadrScaleAttack import QuadraticScaleAttack
from defenses.detection.fourier.FourierPeakMatrixCollector import FourierPeakMatrixCollector, PeakMatrixMethod
from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries
from torch.autograd import Variable
from tqdm import trange

from scaleadv.bypass.random import resize_to_224x
from scaleadv.datasets.imagenet import create_dataset
from scaleadv.models.layers import MedianPool2d, RandomPool2d
from scaleadv.models.parallel import BalancedDataParallel


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
    RUN_ADA = False
    RUN_MEDIAN_DEF = False
    RUN_RANDOM_DEF = True

    if RUN_MEDIAN_DEF:
        defense = 'median'
        filter = MedianPool2d(9, 1, 4)
    elif RUN_RANDOM_DEF:
        defense = 'random'
        filter = BalancedDataParallel(10, RandomPool2d(9, 1, 4))
    else:
        raise NotImplementedError

    # load data
    dataset = create_dataset(transform=None)
    _, src, _ = dataset[5000]
    _, tgt, _ = dataset[1000]
    src = resize_to_224x(src, more=2)
    x_src = np.array(src)
    x_tgt = np.array(tgt)

    # load scaling & scaled target image
    lib = SuppScalingLibraries.CV
    algo = SuppScalingAlgorithms.AREA
    scaling = ScalingGenerator.create_scaling_approach(x_src.shape, (224, 224, 4), lib, algo)
    x_tgt = scaling.scale_image(x_tgt)

    # get modified pixel's mask
    cl, cr = scaling.cl_matrix, scaling.cr_matrix
    cli, cri = map(LA.pinv, [cl, cr])
    mask = np.round(cli @ np.ones((224, 224)) @ cri).astype(np.uint8)

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
    diff_l1 = nn.L1Loss()
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
        Image.fromarray(att).save(f'{TAG}.attack.png')
        Image.fromarray(att_inp).save(f'{TAG}.attack-inp.png')

    """The following snippet implements adaptive scaling-attack with torch
    """
    if RUN_ADA:
        n = 1 if defense == 'median' else 70
        att_proxy = Variable(torch.zeros_like(src), requires_grad=True)
        mask_t = torch.tensor(mask, dtype=torch.float32).to(att_proxy.device)
        optimizer = torch.optim.Adam([att_proxy], lr=0.01)
        with trange(100) as pbar:
            for _ in pbar:
                att = (att_proxy.tanh() + 1) * 0.5
                att_def = att * (1 - mask_t) + filter(att.repeat(n, 1, 1, 1)) * mask_t
                out = model(att_def)
                loss1 = diff(src, att)
                loss2 = diff_l1(tgt, out)
                loss3 = torch.tensor(0)#0.1 * (get_color(src - att) + get_sim(src - att))
                loss = loss1 + loss2 + loss3
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix({
                    'SRC': f'{loss1.cpu().item():.3f}',
                    'OUT': f'{loss2.cpu().item():.3f}',
                    'COLOR': f'{loss3.cpu().item():.3f}',
                })

        # save results
        F.to_pil_image(att.cpu()).save(f'{TAG}.adaptive.png')



    """The following snippet implements median-defense with torch
    """
    att_name = f'{TAG}.attack'
    att = Image.open(f'{att_name}.png')
    x = F.to_tensor(att)[None, ...]
    filter = RandomPool2d(9, 1, 4)

    # global median filter
    x_o = F.to_pil_image(filter(x)[0])
    x_o.save(f'{att_name}.global-{defense}.png')
    x_o_inp = scaling.scale_image(np.array(x_o))
    Image.fromarray(x_o_inp).save(f'{att_name}.global-{defense}-inp.png')

    # selective median filter
    x_o = x * (1 - mask) + filter(x) * mask
    x_o = F.to_pil_image(x_o[0].float())
    x_o.save(f'{att_name}.select-{defense}.png')
    x_o_inp = scaling.scale_image(np.array(x_o))
    Image.fromarray(x_o_inp).save(f'{att_name}.select-{defense}-inp.png')

    # selective median filter with mask-out
    mask_l, mask_h = mask.copy(), mask.copy()
    mask_l[0::2, 0::2] = mask_l[1::2, 1::2] = 0
    mask_h[1::2, 0::2] = mask_h[0::2, 1::2] = 0

    x_f = x.clone()
    x_f[(slice(None),) * 2 + mask_l.nonzero()] = 0
    x_f[(slice(None),) * 2 + mask_h.nonzero()] = 1
    x_f = filter(x_f)
    x_o = x * (1 - mask) + x_f * mask
    x_o = F.to_pil_image(x_o[0].float())
    x_o.save(f'{att_name}.select-{defense}-mask.png')
    x_o_inp = scaling.scale_image(np.array(x_o))
    Image.fromarray(x_o_inp).save(f'{att_name}.select-{defense}-mask-inp.png')
