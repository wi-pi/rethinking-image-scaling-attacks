import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
from art.estimators.classification import PyTorchClassifier
from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries

from scaleadv.attacks.adv import IndirectPGD
from scaleadv.attacks.scale_nn import ScaleAttack
from scaleadv.datasets.imagenet import create_dataset
from scaleadv.models.layers import NormalizationLayer
from scaleadv.models.resnet import resnet50_imagenet
from scaleadv.models.scaling import ScaleNet
from scaleadv.tests.utils import resize_to_224x

TAG = 'TEST.ScaleAttack.Adv'

if __name__ == '__main__':
    # set params
    norm, sigma, step = 2, 20, 30
    epsilon = sigma * 2.5 / step
    target = 200

    # load data
    dataset = create_dataset(transform=None)
    src, _ = dataset[5000]
    src = resize_to_224x(src)
    src = np.array(src)

    # load scaling
    lib = SuppScalingLibraries.CV
    algo = SuppScalingAlgorithms.LINEAR
    scaling = ScalingGenerator.create_scaling_approach(src.shape, (224, 224, 4), lib, algo)

    # load scaled src image & to tensor
    src_inp = scaling.scale_image(src)
    src_inp = F.to_tensor(src_inp).numpy()[None, ...]
    src = F.to_tensor(src).numpy()[None, ...]

    # load network & attack
    scale_net = ScaleNet(scaling.cl_matrix, scaling.cr_matrix).eval().cuda()
    class_net = nn.Sequential(NormalizationLayer.from_preset('imagenet'), resnet50_imagenet('2')).eval().cuda()
    classifier = PyTorchClassifier(class_net, nn.CrossEntropyLoss(), (3, 224, 224), 1000, clip_values=(0, 1))
    adv_attack = IndirectPGD(classifier, norm, sigma, epsilon, step, targeted=True)
    scl_attack = ScaleAttack(scale_net, lam_inp=2)

    # adv attack
    y_target = np.eye(1000, dtype=np.int)[None, target]
    adv = adv_attack.generate(x=src_inp, y=y_target, proxy=None)

    # scale attack
    att, att_inp = scl_attack.generate(src=src, tgt=adv)

    # test adv
    ns = 'SRC', 'ADV', 'ATT'
    vs = src_inp, adv, att_inp
    for n, v in zip(ns, vs):
        print(n, classifier.predict(v).argmax(1)[0])

    # save figs
    f = T.Compose([lambda x: x[0], torch.tensor, T.ToPILImage()])
    for n in ['src', 'src_inp', 'adv', 'att', 'att_inp']:
        var = locals()[n]
        f(var).save(f'{TAG}.{n}.png')
