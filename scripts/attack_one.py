from argparse import ArgumentParser

import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from art.attacks.evasion import ProjectedGradientDescentPyTorch as PGD
from art.estimators.classification import PyTorchClassifier
from loguru import logger

from scaleadv.attacks.core import ScaleAttack
from scaleadv.datasets import get_imagenet
from scaleadv.datasets.transforms import Align
from scaleadv.defenses import POOLING_MAPS
from scaleadv.evaluate import Evaluator
from scaleadv.models import resnet50
from scaleadv.models.resnet import IMAGENET_MODEL_PATH
from scaleadv.scaling import ScalingAPI, ScalingLib, ScalingAlg

NORM = {
    '2': 2,
    'inf': np.inf
}

if __name__ == '__main__':
    p = ArgumentParser()
    _ = p.add_argument
    # Input args
    _('--id', type=int, required=True, help='ID of test image')
    _('--model', default=None, type=str, choices=IMAGENET_MODEL_PATH.keys(), help='use robust model, optional')
    # Scaling args
    _('--lib', default='cv', type=str, choices=ScalingLib.names(), help='scaling libraries')
    _('--alg', default='linear', type=str, choices=ScalingAlg.names(), help='scaling algorithms')
    _('--scale', default=None, type=int, help='set a fixed scale ratio, unset to use the original size')
    # Adversarial attack args
    _('--norm', default='2', type=str, choices=NORM.keys(), help='adv-attack norm')
    _('--eps', default=20, type=float, help='L2 perturbation of adv-example')
    _('--step', default=30, type=int, help='max iterations of PGD attack')
    # Scaling attack args
    _('--defense', default='none', type=str, choices=POOLING_MAPS.keys(), help='type of defense')
    _('--samples', default=1, type=int, help='number of samples to approximate random pooling')
    _('--std', default=1.0, type=float, help='std of non-uniform random pooling')
    # Misc args
    _('--tag', default='TEST', type=str, help='prefix of names')

    # Sub commands
    sp = p.add_subparsers(dest='action')
    # HIDE args
    p_hide = sp.add_parser('hide', help='ScaleAdv - Hide')
    p_hide.add_argument('--lr', default=0.01, type=float, help='learning rate for scaling attack')
    p_hide.add_argument('--weight', default=2, type=int, help='weight for MSE at the input space')
    p_hide.add_argument('--iter', default=100, type=int, help='max iterations of Scaling attack')
    # GENERATE args
    p_gen = sp.add_parser('generate', help='ScaleAdv - Generate')
    p_gen.add_argument('--big-eps', default=40, type=float, help='L2 perturbation of attack image')
    p_gen.add_argument('--big-sig', default=4.0, type=float, help='L2 perturbation step size')
    p_gen.add_argument('--big-step', default=100, type=int, help='max iterations of Scale-Adv')
    args = p.parse_args()
    args.norm = NORM[args.norm]

    # Load data
    transform = T.Compose([Align(224, args.scale), T.ToTensor(), lambda x: np.array(x)[None, ...]])
    dataset = get_imagenet('val', transform)
    src, y_src = dataset[args.id]
    logger.info(f'Loading source image: id {args.id}, label {y_src}, shape {src.shape}, dtype {src.dtype}.')

    # Load scaling
    scaling_api = ScalingAPI(src.shape[-2:], (224, 224), args.lib, args.alg)
    inp = scaling_api(src[0])[None, ...]

    # Load pooling
    cls = POOLING_MAPS[args.defense]
    prob_kwargs = {'std': args.std}
    pooling_layer = cls.auto(round(scaling_api.ratio) * 2 - 1, scaling_api.mask).cuda()

    # Load network
    class_network = resnet50(args.model, normalize=True).eval().cuda()

    # Prepare adv attack
    classifier = PyTorchClassifier(class_network, nn.CrossEntropyLoss(), inp.shape[1:], 1000, clip_values=(0, 1))

    # Adv Attack
    norm, eps, max_iter = args.norm, args.eps, args.step
    eps_step = eps * 2.5 / max_iter
    logger.info(f'Loading PGD attack: norm {norm}, eps {eps:.3f}, eps_step {eps_step:.3f}, max_iter {max_iter}.')
    adv_attack = PGD(classifier, norm, eps, eps_step, max_iter, verbose=False)
    adv = adv_attack.generate(inp, np.array([y_src]))

    # Initial test
    y_inp = classifier.predict(inp).argmax(1).item()
    y_adv = classifier.predict(adv).argmax(1).item()
    logger.info(f'Initial prediction: inp = {y_inp}.')
    logger.info(f'Initial prediction: adv = {y_adv}.')

    # Scaling attack
    attack = ScaleAttack(scaling_api, pooling_layer, class_network)
    if args.action == 'hide':
        att = attack.hide(src, adv, y_src, y_adv, args.iter, args.lr, args.weight, args.samples)
    elif args.action == 'generate':
        attack_kwargs = dict(norm=args.norm, eps=args.big_eps, eps_step=args.big_sig, max_iter=args.big_step)
        att = attack.generate(src, y_src, PGD, attack_kwargs)
    else:
        raise NotImplementedError

    # Evaluate
    e = Evaluator(scaling_api, class_network)
    e.eval(src, adv, att, y_src, y_adv, tag=f'test.{args.id}.{args.action}', show=True, save='.')
