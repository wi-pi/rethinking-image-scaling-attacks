import sys
import torch
import numpy as np
from PIL import Image

from scaleadv import metric as M


def load(filename):
    return np.array(Image.open(filename))

def to_tensor(x):
    x = x.transpose(2, 0, 1)
    x = torch.as_tensor(x / 255, dtype=torch.float32)
    x = x[None, ...]
    return x.clamp(0, 1)


def compare(x, y, cap='Compare'):
    # now x, y are ndarray[WHC] in [0, 1]
    x, y = map(to_tensor, [x, y])
    print(cap, end=': ')

    # norms
    print(f'[L2] {M.L2(x, y):.5f}', end='\t')
    print(f'[Linf] {M.Linf(x, y):.5f}', end='\t')

    # perceptual
    print(f'[PSNR] {M.psnr(x, y):.5f}', end='\t')
    print(f'[SSIM] {M.ssim(x, y):.5f}', end='\t')
    print(f'[MS-SSIM] {M.msssim(x, y):.5f}', end='\t')
    print(f'[LPIPS] {M.lpips(x, y):.5f}', end='\t')
    print()


def run(test_id):
    # small
    x_src = load(f'static/results/scaleadv/{test_id}.src.small.png')
    x_adv = load(f'static/results/scaleadv/{test_id}.x_adv_scl.small.png')
    x_scl = load(f'static/results/scaleadv/{test_id}.x_ada_defense.small.png')
    compare(x_src, x_adv, f'Adv.inp')
    compare(x_src, x_scl, f'Scl.inp')

    # big
    x_src = load(f'static/results/scaleadv/{test_id}.src.png')
    x_adv = load(f'static/results/scaleadv/{test_id}.x_adv_scl.big.png')
    x_scl = load(f'static/results/scaleadv/{test_id}.x_ada_defense.big.png')
    compare(x_src, x_adv, 'Adv.big')
    compare(x_src, x_scl, 'Scl.big')


if __name__ == '__main__':
    run(int(sys.argv[1]))
