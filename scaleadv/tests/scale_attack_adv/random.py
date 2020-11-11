import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
from art.attacks.evasion import CarliniL2Method, ShadowAttack
from art.estimators.classification import PyTorchClassifier
from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries
from torch.nn import DataParallel

from scaleadv.attacks.adv import IndirectPGD, SamplePGD
from scaleadv.attacks.proxy import PoolingProxy, NoiseProxy
from scaleadv.attacks.scale_nn import ScaleAttack
from scaleadv.attacks.shadow import SmoothAttack
from scaleadv.attacks.utils import get_mask_from_cl_cr
from scaleadv.datasets.imagenet import create_dataset
from scaleadv.models.layers import NormalizationLayer, MedianPool2d, RandomPool2d
from scaleadv.models.parallel import BalancedDataParallel
from scaleadv.models.resnet import resnet50_imagenet
from scaleadv.models.scaling import ScaleNet
from scaleadv.tests.utils import resize_to_224x, Evaluator

if __name__ == '__main__':
    # set params
    norm, sigma, step = 2, 20, 30
    epsilon = sigma * 2.5 / step
    target = 200
    ID = 5000
    TAG = f'RND-{ID}.Sig{sigma}'

    # load data
    dataset = create_dataset(transform=None)
    src, _ = dataset[ID]
    src = resize_to_224x(src)
    src = np.array(src)

    # load scaling
    lib = SuppScalingLibraries.CV
    algo = SuppScalingAlgorithms.LINEAR
    scaling = ScalingGenerator.create_scaling_approach(src.shape, (224, 224, 3), lib, algo)
    mask = get_mask_from_cl_cr(scaling.cl_matrix, scaling.cr_matrix)

    # load scaled src image & to tensor
    src_inp = scaling.scale_image(src)
    src_inp = F.to_tensor(src_inp).numpy()[None, ...]
    src = F.to_tensor(src).numpy()[None, ...]

    # load network & attack
    pooling = RandomPool2d(5, 1, 2, mask)
    scale_net = ScaleNet(scaling.cl_matrix, scaling.cr_matrix).eval().cuda()
    class_net = nn.Sequential(NormalizationLayer.from_preset('imagenet'), resnet50_imagenet('2')).eval()
    class_net = BalancedDataParallel(12, class_net).cuda()
    classifier = PyTorchClassifier(class_net, nn.CrossEntropyLoss(), (3, 224, 224), 1000, clip_values=(0, 1))
    adv_attack = IndirectPGD(classifier, norm, sigma, epsilon, step, targeted=True, batch_size=300)
    # adv_attack = CarliniL2Method(classifier, confidence=3.0, targeted=True, binary_search_steps=20, max_iter=20)
    # adv_attack = ShadowAttack(classifier, sigma=0.1, targeted=True, batch_size=300)
    scl_attack = ScaleAttack(scale_net, class_net, pooling, lr=0.1, lam_inp=200, nb_samples=200, max_iter=120, early_stop=True)   # for None proxy
    # scl_attack = ScaleAttack(scale_net, class_net, pooling, lr=0.05, lam_inp=40, nb_samples=200, max_iter=200, early_stop=True)  # for noise proxy

    # adv attack
    """
    Note:
        1. NoiseProxy with sigma = 0.1 also works. (makes scale-attack more consistent)
        2. being robust to random-filter is like robust to normal/laplacian noise
        3. This is left for future experiments, ideally we don't rely on proxy.
    """
    # proxy = PoolingProxy(pooling, n=300, x_big=src, scale=scale_net)
    # proxy = NoiseProxy(np.random.normal, n=300, loc=0, scale=0.1)
    y_target = np.eye(1000, dtype=np.int)[None, target]
    adv = adv_attack.generate(x=src_inp, y=y_target, proxy=None)

    # scale attack
    att = scl_attack.generate(src=src, tgt=adv, adaptive=True, mode='sample', test_freq=10)
    # att = scl_attack.generate_optimal(src=src, target=target)

    # test
    E = Evaluator(scale_net, class_net, (5, 1, 2, mask))
    E.eval(src, adv, att, summary=True, tag=TAG, save='.')