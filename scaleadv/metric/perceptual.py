import torch
import piq
from lpips import LPIPS


def psnr(x, y):
    return piq.psnr(x, y, data_range=1., reduction='none').item()


def ssim(x, y):
    return piq.ssim(x, y, data_range=1.).item()


def msssim(x, y):
    return piq.multi_scale_ssim(x, y, data_range=1.).item()


def lpips(x, y):
    x = x * 2 - 1
    y = y * 2 - 1
    return LPIPS(net='alex', verbose=False)(x, y).item()
