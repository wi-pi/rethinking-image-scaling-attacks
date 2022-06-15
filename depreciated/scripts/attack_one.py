from argparse import ArgumentParser

import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from art.attacks.evasion import ProjectedGradientDescentPyTorch as PGD
from art.estimators.classification import PyTorchClassifier
from loguru import logger
from torch.nn import DataParallel

from depreciated.scaleadv import ScaleAttack
from depreciated.scaleadv import get_imagenet
from depreciated.scaleadv import Align
from depreciated.scaleadv import POOLING_MAPS
from depreciated.scaleadv.evaluate import Evaluator
from depreciated.scaleadv import resnet50
from depreciated.scaleadv import IMAGENET_MODEL_PATH
from depreciated.scaleadv import ScalingAPI, str_to_alg, str_to_lib

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
    _('--lib', default='cv', type=str, choices=str_to_lib.keys(), help='scaling libraries')
    _('--alg', default='linear', type=str, choices=str_to_alg.keys(), help='scaling algorithms')
    _('--scale', default=None, type=int, help='set a fixed scale ratio, unset to use the original size')
    # Adversarial attack args
    _('--norm', default='2', type=str, choices=NORM.keys(), help='adv-attack norm')
    _('--eps', default=20, type=float, help='L2 perturbation of adv-example')
    _('--step', default=30, type=int, help='max iterations of PGD attack')
    # Scaling attack args
    _('--defense', default='none', type=str, choices=POOLING_MAPS.keys(), help='type of defense')
    _('--samples', default=1, type=int, help='number of samples to approximate random pooling')
    _('--cache', default=20, type=int, help='number of iters to cache noise')
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
    transform = T.Compose([Align(256, args.scale), T.ToTensor(), lambda x: np.array(x)[None, ...]])
    dataset = get_imagenet('val', transform)
    src, y_src = dataset[args.id]
    logger.info(f'Loading source image: id {args.id}, label {y_src}, shape {src.shape}, dtype {src.dtype}.')

    # Load scaling
    scaling_api = ScalingAPI(src.shape[-2:], (256, 256), args.lib, args.alg)
    inp = scaling_api(src[0])[None, ...]

    # Load pooling
    cls = POOLING_MAPS[args.defense]
    prob_kwargs = {'std': args.std}
    pooling_layer = cls.auto(round(scaling_api.ratio) * 2 - 1, scaling_api.mask).cuda()

    # Load network
    class_network = resnet50(args.model, normalize=True).eval()
    if args.samples > 90:
        class_network = DataParallel(class_network)
    class_network = class_network.cuda()

    # Prepare adv attack
    classifier = PyTorchClassifier(class_network, nn.CrossEntropyLoss(), inp.shape[1:], 1000, clip_values=(0, 1))

    # Adv Attack
    norm, eps, max_iter = args.norm, args.eps, args.step
    eps_step = eps * 10 / max_iter
    logger.info(f'Loading PGD attack: norm {norm}, eps {eps:.3f}, eps_step {eps_step:.3f}, max_iter {max_iter}.')
    adv_attack = PGD(classifier, norm, eps, eps_step, max_iter, targeted=False, verbose=False)
    # adv_attack = CarliniL2Method(classifier, confidence=args.eps, binary_search_steps=30, max_iter=20, targeted=True)
    adv = adv_attack.generate(inp, np.array([y_src]))

    # Initial test
    y_inp = classifier.predict(inp).argmax(1).item()
    y_adv = classifier.predict(adv).argmax(1).item()
    logger.info(f'Initial prediction: inp = {y_inp}.')
    logger.info(f'Initial prediction: adv = {y_adv}.')

    # Scaling attack
    attack = ScaleAttack(scaling_api, pooling_layer, class_network, nb_samples=args.samples, nb_flushes=args.cache, verbose=True)
    if args.action == 'hide':
        att = attack.hide(src, adv, y_src, y_adv, args.iter, args.lr, args.weight, verbose=True)
    elif args.action == 'generate':
        attack_kwargs = dict(norm=args.norm, eps=args.big_eps, eps_step=args.big_sig, max_iter=args.big_step)
        # attack_kwargs = dict(confidence=args.eps, binary_search_steps=30, max_iter=20, targeted=True)
        logger.info('Start generating...')
        # att = attack.generate(src, 90, CarliniL2Method, attack_kwargs)
        att = attack.generate(src, y_src, PGD, attack_kwargs)
        logger.info('Done generating...')
    else:
        raise NotImplementedError

    # Evaluate
    e = Evaluator(scaling_api, class_network, nb_samples=50)
    e.eval(src, adv, att, y_src, y_adv, tag=f'test.{args.id}.{args.action}.{args.defense}', show=True, save='.')
