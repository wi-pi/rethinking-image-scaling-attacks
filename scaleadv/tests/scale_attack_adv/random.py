import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
from art.attacks.evasion import CarliniL2Method
from art.estimators.classification import PyTorchClassifier
from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries
from torch.nn import DataParallel

from scaleadv.attacks.adv import IndirectPGD
from scaleadv.attacks.proxy import PoolingProxy
from scaleadv.attacks.scale_nn import ScaleAttack
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
    scl_attack = ScaleAttack(scale_net, pooling, class_net, lr=0.1, lam_inp=40, nb_samples=300, max_iter=200, early_stop=True)

    # attack src_def instead
    if attack_pooling:
        x = pooling(torch.tensor(src).cuda())
        src_inp = scale_net(x).cpu().numpy()
        # src = x.cpu().numpy()

    # adv attack
    proxy = PoolingProxy(pooling, n=300, x_big=src, scale=scale_net)
    y_target = np.eye(1000, dtype=np.int)[None, target]
    adv = adv_attack.generate(x=src_inp, y=y_target, proxy=proxy)
    print(f'ADV', classifier.predict(adv).argmax(1))

    # scale attack
    att, att_inp = scl_attack.generate(src=src, tgt=adv, use_pooling=True, use_ce=False, y_tgt=target)
    # att, att_inp = scl_attack.generate_test(src=src, tgt=adv, y_tgt=target)

    # test adv
    ns = 'src', 'adv', 'att'
    vs = src, adv, att
    bs = True, False, True
    for n, v, b in zip(ns, vs, bs):
        # to tensor
        v = torch.tensor(v).cuda()
        # no pooling
        x = scale_net(v).cpu() if b else v.cpu()
        print(n + ' ', classifier.predict(x).argmax(1)[0])
        # big and pooling
        if b:
            x = scale_net(pooling(v.cpu()).cuda()).cpu()
            print(n + '*', classifier.predict(x).argmax(1)[0])

    # save figs
    f = T.Compose([lambda x: x[0], torch.tensor, T.ToPILImage()])
    for n in ['src', 'src_inp', 'adv', 'att', 'att_inp']:
        var = locals()[n]
        f(var).save(f'{TAG}.{n}.png')
