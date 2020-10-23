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

class ScalingNet(nn.Module):

    def __init__(self, cl: np.ndarray, cr: np.ndarray):
        super(ScalingNet, self).__init__()
        self.cl = nn.Parameter(torch.as_tensor(cl.copy(), dtype=torch.float32), requires_grad=False)
        self.cr = nn.Parameter(torch.as_tensor(cr.copy(), dtype=torch.float32), requires_grad=False)

    def forward(self, inp: torch.Tensor):
        return self.cl @ inp @ self.cr


def predict(inp):
    if isinstance(inp, torch.Tensor):
        inp = inp.detach().cpu()
    return classifier.predict(inp).argmax(1)


def l2_loss(x, y):
    return (x - y).norm((1, 2, 3)).mean()


if __name__ == '__main__':
    TAG = 'ADA'
    RUN_ADV = True
    RUN_CVX = False
    RUN_SGD = True
    RUN_ADA = True
    RUN_MEDIAN_DEF = True
    RUN_RANDOM_DEF = False

    if RUN_MEDIAN_DEF:
        defense = 'median'
        filter = MedianPool2d(5, 1, 2)
    elif RUN_RANDOM_DEF:
        defense = 'random'
        # filter = BalancedDataParallel(10, RandomPool2d(7, 1, 3))
        filter = RandomPool2d(5, 1, 2)
    else:
        raise NotImplementedError

    # load data
    dataset = create_dataset(transform=None)
    _, src, _ = dataset[5000]
    _, tgt, _ = dataset[1000]
    src = resize_to_224x(src, more=1)
    x_src = np.array(src)
    x_tgt = np.array(tgt)

    # load scaling & scaled target image
    lib = SuppScalingLibraries.CV
    algo = SuppScalingAlgorithms.LINEAR
    scaling = ScalingGenerator.create_scaling_approach(x_src.shape, (224, 224, 4), lib, algo)
    x_tgt = scaling.scale_image(x_tgt)

    # save src
    src.save(f'{TAG}.src.png')
    Image.fromarray(scaling.scale_image(x_src)).save(f'{TAG}.src-inp.png')

    # get modified pixel's mask
    cl, cr = scaling.cl_matrix, scaling.cr_matrix
    cli, cri = map(LA.pinv, [cl, cr])
    mask = np.round(cli @ np.ones((224, 224)) @ cri).astype(np.uint8)

    # load art proxy
    model_cls = nn.Sequential(
        NormalizationLayer(IMAGENET_MEAN, IMAGENET_STD),
        get_model(weights_file=MODEL_PATH[2]),
        # resnet50(pretrained=True)
    ).eval().cuda()
    classifier = PyTorchClassifier(model_cls, nn.CrossEntropyLoss(), (3, 224, 224), 1000, clip_values=(0., 1.))

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

    """The following snippet implements adv-attack with art
    """
    # EXPERIMENT: Attack filtered src
    x = x_src
    x_src = F.to_pil_image(filter(F.to_tensor(x)[None, ...])[0])
    x_src.save(f'{TAG}.src-def.png')
    x_src = np.array(x_src)
    if RUN_ADV:
        # NORM, SIGMA, STEP = np.inf, 16 / 255, 30
        NORM, SIGMA, STEP = 2, 20, 30
        attack = ProjectedGradientDescentPyTorch(classifier, NORM, SIGMA, SIGMA * 2.5 / STEP, max_iter=STEP,
                                                 targeted=True)
        x = F.to_tensor(scaling.scale_image(x))[None, ...]
        print('y_pred', predict(x))
        y_tgt = np.eye(1000, dtype=np.int)[None, 200]
        x_adv = attack.generate(x, y_tgt)
        print('y_adv', predict(x_adv))
        Image.fromarray(np.array(x_adv[0] * 255).round().astype(np.uint8).transpose((1, 2, 0))).save(f'{TAG}.adv.png')

    """The following snippet implements scaling-attack with torch
    """
    # load network
    model = ScalingNet(scaling.cl_matrix, scaling.cr_matrix).eval().cuda()
    diff = nn.MSELoss(reduction='mean')
    diff_l1 = nn.L1Loss(reduction='mean')
    src, tgt = map(lambda x: F.to_tensor(x).cuda(), [x_src, x_tgt])
    if RUN_ADV:
        tgt = torch.tensor(x_adv[0], dtype=torch.float32).cuda()

    # attack
    if RUN_SGD:
        print('SRC:', src.shape, src.min().cpu().item(), src.max().cpu().item())
        print('TGT:', tgt.shape, tgt.min().cpu().item(), tgt.max().cpu().item())
        att_proxy = Variable(src.clone().detach(), requires_grad=True)
        att_proxy.data = ((att_proxy.data * 2 - 1) * (1 - 1e-6)).atanh()
        optimizer = torch.optim.Adam([att_proxy], lr=0.01)
        with trange(1000) as pbar:
            for _ in pbar:
                att = (att_proxy.tanh() + 1) * 0.5
                out = model(att)
                loss1 = (src - att).norm(2)
                loss2 = (tgt - out).norm(2)
                loss = loss1 + 2 * loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix({
                    'SRC': f'{loss1.cpu().item():.3f}',
                    'OUT': f'{loss2.cpu().item():.3f}',
                    # 'PRED': f'{model_cls(out[None, ...]).argmax(1).cpu().item()}'
                })

        # save torch results
        att = np.array(att.detach().cpu() * 255).round().astype(np.uint8).transpose((1, 2, 0))
        att_inp = scaling.scale_image(att)
        Image.fromarray(att).save(f'{TAG}.attack.png')
        Image.fromarray(att_inp).save(f'{TAG}.attack-inp.png')
        print('y_att', predict(F.to_tensor(att_inp)[None, ...]))

    """The following snippet implements adaptive scaling-attack with torch
    """
    if RUN_ADA:
        n = 1 if defense == 'median' else 256
        T = 1000 if defense == 'median' else 50
        att_proxy = Variable(src.clone().detach(), requires_grad=True)
        att_proxy.data = ((att_proxy.data * 2 - 1) * (1 - 1e-6)).atanh()
        mask_t = torch.tensor(mask, dtype=torch.float32).to(att_proxy.device)
        optimizer = torch.optim.Adam([att_proxy], lr=0.3)  # median
        # optimizer = torch.optim.Adam([att_proxy], lr=0.5)  # random
        y_tgt = torch.LongTensor([200]).repeat((n,)).to(device=att_proxy.device)
        with trange(T) as pbar:
            for _ in pbar:
                # get defensed image (big)
                att = (att_proxy.tanh() + 1) * 0.5
                att_def = filter(att.repeat(n, 1, 1, 1))  # use att.cpu if random-defense
                att_def = att * (1 - mask_t) + att_def.cuda() * mask_t
                att_def = att_def.cuda()
                # get scaled image (small)
                out = model(att_def)
                # get prediction (small)
                pred = model_cls(out)
                # compute loss
                loss_big = (src - att).reshape(1, -1).norm(2, dim=1).mean()
                loss_inp = (out - tgt).reshape(n, -1).norm(2, dim=1).mean()
                loss_cls = nn.functional.cross_entropy(pred, y_tgt, reduction='mean')
                loss = loss_big + loss_inp + loss_cls
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    pred = model_cls(out).argmax(1).cpu().numpy()
                pbar.set_postfix({
                    'BIG': f'{loss_big.cpu().item():.3f}',
                    'INP': f'{loss_inp.cpu().item():.3f}',
                    'CLS': f'{loss_cls.cpu().item():.3f}',
                    'PRED-100': f'{(pred == 100).mean():.2%}',
                    'PRED-200': f'{(pred == 200).mean():.2%}',
                    # 'PRED': f'{model_cls(out[:1]).argmax(1).cpu().item()}'
                })

        # save results
        F.to_pil_image(att.cpu()).save(f'{TAG}.adaptive.png')
        F.to_pil_image(out[0].cpu()).save(f'{TAG}.adaptive-inp.png')
        F.to_pil_image(att_def[0].detach().cpu()).save(f'{TAG}.adaptive.{defense}.png')
        print('y_ada', predict(out))

    """The following snippet implements median-defense with torch
    """
    for target in ['attack', 'adaptive']:
        att_name = f'{TAG}.{target}'
        att = Image.open(f'{att_name}.png')
        x = F.to_tensor(att)[None, ...]

        # global median filter
        x_o = F.to_pil_image(filter(x)[0].cpu())
        x_o.save(f'{att_name}.global-{defense}.png')
        x_o_inp = scaling.scale_image(np.array(x_o))
        Image.fromarray(x_o_inp).save(f'{att_name}.global-{defense}-inp.png')
        print('y_def_global', predict(F.to_tensor(x_o_inp)[None, ...]))

        # selective median filter
        x_o = x * (1 - mask) + filter(x).cpu() * mask
        x_o = F.to_pil_image(x_o[0].float())
        x_o.save(f'{att_name}.select-{defense}.png')
        x_o_inp = scaling.scale_image(np.array(x_o))
        Image.fromarray(x_o_inp).save(f'{att_name}.select-{defense}-inp.png')
        print('y_def_select', predict(F.to_tensor(x_o_inp)[None, ...]))

        # # selective median filter with mask-out
        # mask_l, mask_h = mask.copy(), mask.copy()
        # mask_l[0::2, 0::2] = mask_l[1::2, 1::2] = 0
        # mask_h[1::2, 0::2] = mask_h[0::2, 1::2] = 0
        #
        # x_f = x.clone()
        # x_f[(slice(None),) * 2 + mask_l.nonzero()] = 0
        # x_f[(slice(None),) * 2 + mask_h.nonzero()] = 1
        # x_f = filter(x_f).cpu()
        # x_o = x * (1 - mask) + x_f * mask
        # x_o = F.to_pil_image(x_o[0].float())
        # x_o.save(f'{att_name}.select-{defense}-mask.png')
        # x_o_inp = scaling.scale_image(np.array(x_o))
        # Image.fromarray(x_o_inp).save(f'{att_name}.select-{defense}-mask-inp.png')
        # print('y_def_select_mask', predict(F.to_tensor(x_o_inp)[None, ...]))
