import torch
import piq
import lpips


def psnr(x, y):
    return piq.psnr(x, y, data_range=1., reduction='none')


def ssim(x, y):
    return piq.ssim(x, y, data_range=1.)


def msssim(x, y):
    return piq.multi_scale_ssim(x, y, data_range=1.)


def lpips(x, y):
    x = x * 2 - 1
    y = y * 2 - 1
    return lpips.LPIPS(net='alex')(x, y)
