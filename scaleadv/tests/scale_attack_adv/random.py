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

from scaleadv.attacks.adv import IndirectPGD
from scaleadv.attacks.proxy import PoolingProxy, NoiseProxy
from scaleadv.attacks.scale_nn import ScaleAttack
from scaleadv.attacks.shadow import SmoothAttack
from scaleadv.attacks.utils import get_mask_from_cl_cr
from scaleadv.datasets.imagenet import create_dataset
from scaleadv.models.layers import NormalizationLayer, MedianPool2d, RandomPool2d
from scaleadv.models.parallel import BalancedDataParallel
from scaleadv.models.resnet import resnet50_imagenet
from scaleadv.models.scaling import ScaleNet
from scaleadv.tests.utils import resize_to_224x

TAG = 'TEST.ScaleAttack.Adv.Random'

if __name__ == '__main__':
    # set params
    norm, sigma, step = 2, 20, 30
    epsilon = sigma * 2.5 / step
    target = 200
    attack_pooling = False

    # load data
    dataset = create_dataset(transform=None)
    src, _ = dataset[5000]
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
    # scl_attack = ScaleAttack(scale_net, class_net, pooling, lr=0.05, lam_inp=200, nb_samples=200, max_iter=200, early_stop=True)   # for None proxy
    # scl_attack = ScaleAttack(scale_net, class_net, pooling, lr=0.05, lam_inp=40, nb_samples=200, max_iter=200, early_stop=True)  # for noise proxy
    scl_attack = ScaleAttack(scale_net, class_net, pooling, lr=0.05, lam_inp=100, nb_samples=100, max_iter=400, early_stop=True)   # for None proxy and average fintune

    # attack src_def instead
    if attack_pooling:
        x = pooling(torch.tensor(src).cuda())
        src_inp = scale_net(x).cpu().numpy()
        # src = x.cpu().numpy()

    # adv attack
    """
    Note:
        1. NoisePooling with sigma = 0.1 also works. (makes scale-attack more consistent)
        2. being robust to random-filter is like robust to normal/laplacian noise
    """
    # proxy = PoolingProxy(pooling, n=300, x_big=src, scale=scale_net)
    proxy = NoiseProxy(np.random.normal, n=300, loc=0, scale=0.1)
    y_target = np.eye(1000, dtype=np.int)[None, target]
    adv = adv_attack.generate(x=src_inp, y=y_target, proxy=None)
    print(f'ADV', classifier.predict(adv).argmax(1))

    # scale attack
    att = scl_attack.generate(src=src, tgt=adv, adaptive=True, mode='sample', test_freq=10)
    # att = nn.functional.interpolate(torch.tensor(adv), src.shape[2:], mode='bilinear').numpy()
    att_inp = scale_net(pooling(torch.tensor(att)).cuda()).cpu().numpy()

    # test adv
    ns = 'src', 'att'
    vs = src, att
    for n, v in zip(ns, vs):
        # to tensor
        v = torch.tensor(v).cuda()
        pred = scl_attack.predict(v, scale=True, pooling=False)
        print(f'{n}  {np.mean(pred == 100):.2%} {np.mean(pred == 200):.2%}')
        pred = scl_attack.predict(v, scale=True, pooling=True, n=200)
        print(f'{n}* {np.mean(pred == 100):.2%} {np.mean(pred == 200):.2%}')

    # save figs
    f = T.Compose([lambda x: x[0], torch.tensor, T.ToPILImage()])
    for n in ['src', 'src_inp', 'adv', 'att', 'att_inp']:
        var = locals()[n]
        f(var).save(f'{TAG}.{n}.png')
