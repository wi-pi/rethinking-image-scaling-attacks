"""
This module estimates the distribution of perturbation incurred by RandomPool2d.
"""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from loguru import logger
from torch.distributions import Laplace

from depreciated.scaleadv import get_imagenet
from depreciated.scaleadv import Align
from depreciated.scaleadv import RandomPoolingUniform
from depreciated.scaleadv import ScalingAPI, ScalingLib, ScalingAlg
from depreciated.scaleadv.utils import set_ccs_font


def mask_diff(x: torch.Tensor, n: int = 100):
    x = x.repeat(n, 1, 1, 1)
    y = pooling_layer(x)
    diff = (x - y).permute(2, 3, 0, 1)
    diff = diff[scaling_api.mask > 0, ...].flatten()
    return diff


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-i', '--id', type=int, required=True, help='ID of test image')
    p.add_argument('--lib', default='cv', type=str, choices=ScalingLib.names(), help='scaling libraries')
    p.add_argument('--alg', default='linear', type=str, choices=ScalingAlg.names(), help='scaling algorithms')
    p.add_argument('--scale', default=3, type=int, help='scaling ratio')
    args = p.parse_args()

    name = f'static/images/random-dist/dist-{args.scale}.{args.id}'

    # Load data
    transform = T.Compose([Align(224, args.scale), T.ToTensor(), lambda x: x.cuda()[None, ...]])
    dataset = get_imagenet(f'val_{args.scale}', transform)
    src, y_src = dataset[args.id]
    logger.info(f'Loading source image: id {args.id}, label {y_src}, shape {src.shape}, dtype {src.dtype}.')
    F.to_pil_image(src[0].cpu()).save(f'{name}-img.png')

    # Load scaling
    scaling_api = ScalingAPI(src.shape[-2:], (224, 224), args.lib, args.alg)

    # Load pooling
    pooling_layer = RandomPoolingUniform.auto(round(scaling_api.ratio) * 2 - 1, scaling_api.mask).cuda()

    # Sampling
    diff = mask_diff(src, n=100)
    hist = torch.histc(diff, bins=100, min=-1, max=1).cpu().numpy()
    xs = np.arange(-1, 1, 0.02)
    diff = diff.cpu().numpy()

    # Estimate
    med, mad = np.median(diff), np.abs(diff).mean()

    # Plot
    set_ccs_font(17)
    plt.figure(figsize=(7, 6), constrained_layout=True)
    plt.plot(xs, hist / hist.sum(), c='k', lw=1, label=rf'Random Filtering')
    for s in [1.0, 1.5, 2.0]:
        lap = Laplace(loc=med, scale=mad * s).sample([100000]).clamp(-1, 1)
        lap_hist = torch.histc(lap, bins=100, min=-1, max=1).numpy()
        plt.plot(xs, lap_hist / lap_hist.sum(), '--', lw=1, label=rf'Laplace (b={mad:.3f}×{s:.1f})')
    plt.legend(loc='upper left')
    plt.title(rf'Random Filtering ($\hat{{\mu}}$={med:.3f}, $\hat{{b}}$={mad:.3f})')
    plt.savefig(f'{name}.pdf')
