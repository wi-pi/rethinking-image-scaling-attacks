import numpy as np
import torch
import torchvision.transforms as T
from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries

from scaleadv.attacks.scale_nn import ScaleAttack
from scaleadv.datasets.imagenet import create_dataset
from scaleadv.models.layers import MedianPool2d
from scaleadv.models.scaling import ScaleNet
from scaleadv.tests.utils import resize_to_224x

TAG = 'TEST.ScaleAttack.Common.Median'

if __name__ == '__main__':
    # load data
    dataset = create_dataset(transform=None)
    src, _ = dataset[5000]
    tgt, _ = dataset[1000]
    src = resize_to_224x(src)
    src, tgt = map(np.array, [src, tgt])

    # load scaling
    lib = SuppScalingLibraries.CV
    algo = SuppScalingAlgorithms.LINEAR
    scaling = ScalingGenerator.create_scaling_approach(src.shape, (224, 224, 4), lib, algo)
    tgt = scaling.scale_image(tgt)

    # load network
    src = np.array(src / 255, dtype=np.float32).transpose((2, 0, 1))[None, ...]
    tgt = np.array(tgt / 255, dtype=np.float32).transpose((2, 0, 1))[None, ...]
    scale_net = ScaleNet(scaling.cl_matrix, scaling.cr_matrix).eval().cuda()
    pooling = MedianPool2d(5, 1, 2)

    # load attack
    attack = ScaleAttack(scale_net, pooling, lam_inp=4)  # TODO: lam_inp may be relevant to scale factor.
    att, inp = attack.generate(src, tgt, use_pooling=True)

    # save figs
    f = T.Compose([lambda x: x[0], torch.tensor, T.ToPILImage()])
    for n in ['src', 'tgt', 'att', 'inp']:
        var = locals()[n]
        f(var).save(f'{TAG}.{n}.png')
