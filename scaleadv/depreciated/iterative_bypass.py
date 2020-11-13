"""
This module is depreciated due to the following reasons:
    1. directly attack the full pipeline, iterative or not, cannot pass gradients where pixels are not weighted.
    2. ShadowAttack works because the total variation can pass gradients to neighbors.
    3. unless we found a better solution, laplace approximation is the best choice.
"""
import numpy as np
import numpy.linalg as LA
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from RandomizedSmoothing.utils.regularizers import get_tv, get_color, get_sim
from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries
from torch.nn import DataParallel
from tqdm import trange

from scaleadv.datasets.imagenet import create_dataset, IMAGENET_MEAN, IMAGENET_STD
from scaleadv.models.layers import NormalizationLayer
# from scaleadv.experiments.scaling_attack_sgd import ScalingNet
from scaleadv.models.layers import RandomPool2d
from scaleadv.models.scaling import ScaleNet
from scaleadv.depreciated.gen_adv_pgd import get_model
from scaleadv.tests.utils import resize_to_224x

MODEL_PATH = {
    np.inf: 'static/models/imagenet_linf_4.pt',
    2: 'static/models/imagenet_l2_3_0.pt',
}


def test(x: torch.Tensor, n: int):
    with torch.no_grad():
        x = torch.clamp(x, 0, 1)
        xs = pooling(x.cpu().repeat(n, 1, 1, 1)).cuda()
        pred = model(xs).argmax(1)
    return pred.cpu().numpy()


def attack_pgd(x: torch.Tensor, y: int, target: int, fix_pooling: torch.Tensor = None, desc: str = 'Attack'):
    assert len(x.shape) == 3
    if fix_pooling is not None:
        assert len(fix_pooling.shape) == 4

    # To batch
    x = x[None, ...]

    # Prepare attack
    delta = torch.zeros_like(x)
    delta.requires_grad_()

    # att = x + delta
    # fix_pooling = pooling(att.cpu().repeat(N, 1, 1, 1)).cuda()

    sigma, step = 40, 30
    optimizer = torch.optim.SGD([delta], lr=sigma * 2 / step)
    y_tgt = torch.LongTensor([target]).repeat((N,)).to(device=x.device)
    pooling.cache = None
    with trange(step, desc=desc) as pbar:
        for _ in pbar:
            att = x + delta
            att = att.to(pooling.dev)
            att_batch = pooling(att.repeat(N, 1, 1, 1), reuse=True).cuda()
            pred = model(att_batch)
            loss = nn.functional.cross_entropy(pred, y_tgt, reduction='mean').mean()
            optimizer.zero_grad()
            loss.backward()
            old_delta = delta.data.clone()
            optimizer.step()
            # if _ == 10:
            #     print(delta.grad.sum((0,1)).max(0))
            #     from IPython import embed; embed(using=False); exit()
            # clip
            tmp = delta.data.reshape(delta.data.shape[0], -1)
            tmp = tmp * torch.min(
                torch.tensor([1.0], dtype=torch.float32).to(delta.device),
                sigma / (torch.norm(tmp, p=2, dim=1) + 1e-8),
            ).unsqueeze_(-1)
            delta.data = tmp.reshape(delta.data.shape)
            # fix_pooling.data = fix_pooling.data + delta.data - old_delta

            # logging
            pred = pred.argmax(1).cpu().numpy()
            pbar.set_postfix({
                'LOSS': f'{loss.cpu().item():.3f}',
                f'PRED-{y}': f'{(pred == y).mean():.2%}',
                f'PRED-{target}': f'{(pred == target).mean():.2%}',
            })

    return torch.clamp(x + delta, 0, 1).detach()[0]


def attack(x: torch.Tensor, y: int, target: int, fix_pooling: torch.Tensor = None, desc: str = 'Attack'):
    assert len(x.shape) == 3
    if fix_pooling is not None:
        assert len(fix_pooling.shape) == 4
    # prepare optim
    delta = torch.rand_like(x) - 0.5
    delta.requires_grad_()
    optimizer = torch.optim.Adam([delta], lr=0.1)
    y_tgt = torch.LongTensor([target]).repeat((N,)).to(device=x.device)

    # attack iters
    with trange(T, desc=desc) as pbar:
        for _ in pbar:
            # generate current input
            if fix_pooling is None:
                att = x + delta
                att = pooling(att.cpu().repeat(N, 1, 1, 1)).cuda()
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


def iterative_attack(x: torch.Tensor, y: int, target: int):
    att = x.clone().detach()
    for i in range(10):
        # att_batch = pooling(att.cpu().repeat(N, 1, 1, 1)).cuda()
        att = attack_pgd(att, y, target, fix_pooling=None, desc=f'Attack-{i}')
        # test
        F.to_pil_image(att.cpu()).save(f'TEST-{i}.png')
        pred = test(att, n=N)
        print(f'Test {y}: {np.mean(pred == y):.2%}')
        print(f'Test {target}: {np.mean(pred == target):.2%}')
    return att


if __name__ == '__main__':
    # params
    N = 256  # sample numbers
    T = 300  # inner attack iters
    EPS = 1e-6  # tanh
    ID = 5000  # src image
    TGT = 200  # tgt class
    TAG = f'ADA-RANDOM.{ID}'

    # load data & align to 224
    dataset = create_dataset(transform=None)
    src, y_src = dataset[ID]
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
    # pooling = MedianPool2d(7, 1, 3, mask=mask)
    pooling = RandomPool2d(5, 1, 2, mask=mask)
    # pooling = lambda x: x
    model = nn.Sequential(
        # MedianPool2d(7, 1, 3, mask=mask),
        ScaleNet(scaling.cl_matrix, scaling.cr_matrix),
        NormalizationLayer(IMAGENET_MEAN, IMAGENET_STD),
        get_model(weights_file=MODEL_PATH[2]),
        # resnet50(pretrained=True)
    ).eval()
    model = DataParallel(model).cuda()

    # art pgd attack
    # classifier = PyTorchClassifier(model, nn.CrossEntropyLoss(), (3, 672, 672), 1000, clip_values=(0, 1))
    # attack = ProjectedGradientDescentPyTorch(classifier, norm=2, eps=35, eps_step=35*2.5/30, max_iter=30, targeted=True)
    # x = F.to_tensor(src)[None,...]
    # adv = attack.generate(x, np.eye(1000, dtype=np.int)[None, 200])
    # print(classifier.predict(x).argmax(1))
    # print(classifier.predict(adv).argmax(1))
    # Image.fromarray(np.array(adv[0] * 255).round().astype(np.uint8).transpose((1, 2, 0))).save('test.png')
    # from IPython import embed; embed(using=False); exit()

    # N = 1
    # """Direct Attack"""
    # att = attack(F.to_tensor(src).cuda(), y_src, TGT)
    # F.to_pil_image(att.cpu()).save('att-med-res.png')
    # exit()

    """Iterative Attack"""
    att = iterative_attack(F.to_tensor(src).cuda(), y_src, TGT)
    F.to_pil_image(att.cpu()).save('att-iter-ran-rob.png')
    exit()
