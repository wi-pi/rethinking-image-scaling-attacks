import numpy as np
import torch.nn as nn
import torchvision.transforms.functional as F
from art.attacks.evasion import CarliniL2Method
from art.estimators.classification import PyTorchClassifier
from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries

from scaleadv.attacks.adv import IndirectPGD
from scaleadv.attacks.scale_nn import ScaleAttack
from scaleadv.attacks.utils import get_mask_from_cl_cr
from scaleadv.datasets.imagenet import create_dataset
from scaleadv.models.layers import NormalizationLayer, MedianPool2d
from scaleadv.models.resnet import resnet50_imagenet
from scaleadv.models.scaling import ScaleNet
from scaleadv.tests.utils import resize_to_224x, Evaluator

if __name__ == '__main__':
    # set params
    norm, sigma, step = 2, 40, 30
    epsilon = sigma * 2.5 / step
    target = 200
    ID = 5000
    TAG = f'MED-{ID}.Sig{sigma}'

    # load data
    dataset = create_dataset(transform=None)
    src, _ = dataset[ID]
    src = resize_to_224x(src)
    src = np.array(src)

    # load scaling
    lib = SuppScalingLibraries.CV
    algo = SuppScalingAlgorithms.LINEAR
    scaling = ScalingGenerator.create_scaling_approach(src.shape, (224, 224, 4), lib, algo)
    mask = get_mask_from_cl_cr(scaling.cl_matrix, scaling.cr_matrix)

    # load scaled src image & to tensor
    src_inp = scaling.scale_image(src)
    src_inp = F.to_tensor(src_inp).numpy()[None, ...]
    src = F.to_tensor(src).numpy()[None, ...]

    # load network & attack
    pooling = MedianPool2d(5, 1, 2, mask)
    scale_net = ScaleNet(scaling.cl_matrix, scaling.cr_matrix).eval().cuda()
    class_net = nn.Sequential(NormalizationLayer.from_preset('imagenet'), resnet50_imagenet('2')).eval().cuda()
    classifier = PyTorchClassifier(class_net, nn.CrossEntropyLoss(), (3, 224, 224), 1000, clip_values=(0, 1))
    """
    NOTE:
       1. several attacks should work after the median filter, as long as you have a sufficiently large lam_inp.
       2. for high-confidence CW attack, lam_inp can be slightly lower.
       3. currently, scale-attack uses L2 norm in the input space, not sure if we need switch to Linf for Linf-adv.
    """
    adv_attack = CarliniL2Method(classifier, confidence=3.0, targeted=True, binary_search_steps=20, max_iter=20)
    adv_attack = IndirectPGD(classifier, norm, sigma, epsilon, step, targeted=True, batch_size=300)
    scl_attack = ScaleAttack(scale_net, class_net, pooling, lr=0.01, lam_inp=4.0, early_stop=False)

    # adv attack
    y_target = np.eye(1000, dtype=np.int)[None, target]
    adv = adv_attack.generate(x=src_inp, y=y_target)

    # scale attack
    att = scl_attack.generate(src=src, tgt=adv, adaptive=True, include_self=False)

    # test
    E = Evaluator(scale_net, class_net, (5, 1, 2, mask))
    E.eval(src, adv, att, summary=True, tag=TAG, save='.')
