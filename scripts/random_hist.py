"""
This module estimates the distribution of perturbation incurred by RandomPool2d.
"""
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from scaling.ScalingGenerator import ScalingGenerator
from torch.distributions import Laplace

from scaleadv.attacks.utils import get_mask_from_cl_cr, mask_hist
from scaleadv.datasets.imagenet import create_dataset
from scaleadv.models.layers import RandomPool2d
from scaleadv.tests.scale_adv import LIB, ALGO, LIB_TYPE, ALGO_TYPE, INPUT_SHAPE_PIL
from scaleadv.tests.utils import resize_to_224x

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-i', '--id', type=int, required=True, help='ID of test image')
    p.add_argument('--lib', default='cv', type=str, choices=LIB, help='scaling libraries')
    p.add_argument('--algo', default='linear', type=str, choices=ALGO, help='scaling algorithms')
    p.add_argument('--laplace', action='store_true', help='plot laplacian approximation or not')
    args = p.parse_args()

    # Load data
    dataset = create_dataset(transform=None)
    src, _ = dataset[args.id]
    src = resize_to_224x(src, square=True)
    src.save(f'dist-{args.id}-img.png')
    src = np.array(src)

    # Load scaling
    lib = LIB_TYPE[args.lib]
    algo = ALGO_TYPE[args.algo]
    scaling = ScalingGenerator.create_scaling_approach(src.shape, INPUT_SHAPE_PIL, lib, algo)
    mask = get_mask_from_cl_cr(scaling.cl_matrix, scaling.cr_matrix)

    # Convert data to batch ndarray
    normalize_to_batch = T.Compose([T.ToTensor(), lambda x: x.numpy()[None, ...]])
    src_inp = scaling.scale_image(src)
    src, src_inp = map(normalize_to_batch, [src, src_inp])

    # Compute scale ratio
    sr_h, sr_w = [src.shape[i] // src_inp.shape[i] for i in [2, 3]]

    # Load pooling
    k = sr_h * 2 - 1
    pooling = RandomPool2d(k, 1, k // 2, mask)

    # Estimate
    xs, hist, diff = mask_hist(src, pooling, n=100, bins=100)
    mean, std, median, mad = diff.mean(), diff.std(), np.median(diff), np.abs(diff).mean()
    print(f'mean = {mean:.3f}, std = {std:.3f}')
    print(f'median = {median:.3f}, mad = {mad:.3f}')

    # Plot
    plt.figure(constrained_layout=True)
    plt.plot(xs, hist / hist.sum(), c='k', lw=1, label=rf'RandomPooling samples')
    if args.laplace:
        for s in [1.0, 1.5, 2.0]:
            lap = Laplace(loc=median, scale=mad * s).sample([100000]).clamp(-1, 1)
            lap_hist = torch.histc(lap, bins=100, min=-1, max=1).numpy()
            plt.plot(xs, lap_hist / lap_hist.sum(), '--', lw=1, label=rf'Laplace samples ($b$={mad:.3f}*{s:.1f})')
    plt.legend(loc='upper left')
    plt.title(rf'RandomPool2d of ID-{args.id} ($\sigma^2$={std:.3f}, $b$={mad:.3f})')
    tag = '-laplace' if args.laplace else ''
    plt.savefig(f'dist-{args.id}{tag}.pdf')
