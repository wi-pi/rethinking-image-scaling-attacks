"""
This module implements test APIs for Scale-Adv Attack (L2).

Notes:
    1. lam_inp should be sufficiently large to approach a given adv-example.
    2. lam_inp can be relatively smaller for adv-example of high-confidence, including CW and NoisyProxy(scale=0.1).

Empirically good settings for reference:
    1. CW: confidence=3.0, binary_search_steps=20, max_iter=20
    2. Random w/o proxy: lr=0.1, lam_inp=200, max_iter=120
    3. Random w/ proxy: lr=0.05, lam_inp=40, max_iter=200

Scale for images:
    1. 5000: 3*224(0.12), 5x224()
"""
from argparse import ArgumentParser

import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from art.estimators.classification import PyTorchClassifier
from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries

from scaleadv.attacks.adv import IndirectPGD
from scaleadv.attacks.proxy import NoiseProxy
from scaleadv.attacks.scale_nn import ScaleAttack, RANDOM_APPROXIMATION
from scaleadv.attacks.utils import get_mask_from_cl_cr
from scaleadv.datasets.imagenet import IMAGENET_NUM_CLASSES
from scaleadv.datasets.imagenet import create_dataset
from scaleadv.models.layers import NonePool2d
from scaleadv.models.layers import NormalizationLayer, MedianPool2d, RandomPool2d
from scaleadv.models.parallel import BalancedDataParallel
from scaleadv.models.resnet import resnet50_imagenet, IMAGENET_MODEL_PATH
from scaleadv.models.scaling import ScaleNet
from scaleadv.tests.utils import resize_to_224x, Evaluator

# Scaling libs & algorithms
LIB = ['cv', 'tf', 'pil']
ALGO = ['nearest', 'linear', 'cubic', 'lanczos', 'area']
LIB_TYPE = {k: getattr(SuppScalingLibraries, k.upper()) for k in LIB}
ALGO_TYPE = {k: getattr(SuppScalingAlgorithms, k.upper()) for k in ALGO}

# Scaling attack modes
ROBUST_MODELS = IMAGENET_MODEL_PATH.keys()
ADAPTIVE_MODE = RANDOM_APPROXIMATION.keys()
POOLING = {
    'none': NonePool2d,
    'median': MedianPool2d,
    'random': RandomPool2d,
}

# Hard-coded arguments
NUM_CLASSES = IMAGENET_NUM_CLASSES
INPUT_SHAPE_PIL = (224, 224, 3)
INPUT_SHAPE_NP = (3, 224, 224)
FIRST_GPU_BATCH = 16
NUM_SAMPLES_PROXY = 300  # for noisy proxy of adv-attack
NUM_SAMPLES_SAMPLE = 100  # for monte carlo sampling of scale-attack

if __name__ == '__main__':
    p = ArgumentParser()
    # Input args
    p.add_argument('--id', type=int, required=True, help='ID of test image')
    p.add_argument('--target', default=None, type=int, help='target label, unset for un-targeted attack')
    p.add_argument('--model', default=None, type=str, choices=ROBUST_MODELS, help='use robust model, optional')
    # Scaling args
    p.add_argument('--lib', default='cv', type=str, choices=LIB, help='scaling libraries')
    p.add_argument('--algo', default='linear', type=str, choices=ALGO, help='scaling algorithms')
    p.add_argument('--scale', default=0, type=int, help='set a fixed scale ratio, 0 to use the original size')
    # Adversarial attack args
    p.add_argument('--eps', default=20, type=float, help='L2 perturbation of adv-example')
    p.add_argument('--step', default=30, type=int, help='max iterations of PGD attack')
    p.add_argument('--adv-proxy', action='store_true', help='do adv-attack on noisy proxy')
    # Scaling attack args
    p.add_argument('--defense', default='none', type=str, choices=POOLING.keys(), help='type of defense')
    p.add_argument('--mode', default=None, type=str, choices=ADAPTIVE_MODE, help='random pooling approximation mode')
    p.add_argument('--samples', default=1, type=int, help='number of samples to approximate random pooling')
    # Misc args
    p.add_argument('--tag', default='TEST', type=str, help='prefix of names')

    # Sub commands
    sp = p.add_subparsers(dest='action')
    # HIDE args
    p_hide = sp.add_parser('hide', help='ScaleAdv - Hide')
    p_hide.add_argument('--lr', default=0.01, type=float, help='learning rate for scaling attack')
    p_hide.add_argument('--lam-inp', default=1, type=int, help='lambda for L2 penalty at the input space')
    p_hide.add_argument('--iter', default=200, type=int, help='max iterations of Scaling attack')
    # GENERATE args
    p_gen = sp.add_parser('generate', help='ScaleAdv - Generate')
    p_gen.add_argument('--big-eps', default=40, type=float, help='L2 perturbation of attack image')
    p_gen.add_argument('--big-sig', default=4.0, type=float, help='L2 perturbation step size')
    p_gen.add_argument('--big-step', default=30, type=int, help='max iterations of Scale-Adv')
    args = p.parse_args()

    # Load data
    dataset = create_dataset(transform=None)
    src, y_src = dataset[args.id]
    src = resize_to_224x(src, scale=args.scale, square=True)
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

    # Load pooling
    scale_ratio = src.shape[2] // src_inp.shape[2]
    k = scale_ratio * 2 - 1
    pooling_args = (k, 1, k // 2, mask)
    pooling = POOLING[args.defense](*pooling_args)

    # Load networks
    scale_net = ScaleNet(scaling.cl_matrix, scaling.cr_matrix).eval()
    class_net = nn.Sequential(NormalizationLayer.from_preset('imagenet'), resnet50_imagenet(args.model)).eval()
    if args.samples > 1:
        class_net = BalancedDataParallel(FIRST_GPU_BATCH, class_net)

    # Move networks to GPU
    scale_net = scale_net.cuda()
    class_net = class_net.cuda()

    # Add proxy if needed
    proxy = None
    if args.adv_proxy:
        proxy = NoiseProxy(np.random.normal, n=NUM_SAMPLES_PROXY, loc=0, scale=0.1)

    # Adv attack
    classifier = PyTorchClassifier(class_net, nn.CrossEntropyLoss(), INPUT_SHAPE_NP, NUM_CLASSES, clip_values=(0, 1))
    eps_step = 2.5 * args.eps / args.step
    # Set targeted option
    targeted = args.target is not None
    adv_attack = IndirectPGD(classifier, 2, args.eps, eps_step, args.step, targeted, batch_size=NUM_SAMPLES_PROXY)
    y_tgt = np.eye(NUM_CLASSES, dtype=np.int)[None, args.target if targeted else y_src]
    # Generate adv example
    adv = adv_attack.generate(x=src_inp, y=y_tgt, proxy=proxy)

    # Scale attack based on args.action
    scl_attack = ScaleAttack(scale_net, class_net, pooling)
    if args.action == 'hide':
        att = scl_attack.hide(src, adv, lr=args.lr, step=args.iter, lam_inp=args.lam_inp,
                              mode=args.mode, nb_samples=args.samples, attack_self=False,
                              src_label=y_src, tgt_label=args.target, test_freq=0, early_stop=True)
    elif args.action == 'generate':
        attack_args = dict(norm=2, eps=args.big_eps, eps_step=args.big_sig, max_iter=args.big_step, targeted=targeted,
                           batch_size=NUM_SAMPLES_PROXY)
        att = scl_attack.generate(src, y_src, IndirectPGD, attack_args, y_tgt=args.target,
                                  mode=args.mode, nb_samples=args.samples)
    else:
        raise NotImplementedError

    # Test
    e = Evaluator(scale_net, class_net, pooling_args)
    e.eval(src, adv, att, summary=True, y_adv=args.target,
           tag=f'{args.tag}.{args.id}.{args.action}.{args.defense}.{args.mode}',
           save='.')
