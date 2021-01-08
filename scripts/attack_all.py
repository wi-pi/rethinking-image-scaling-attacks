import os
import pickle
from argparse import ArgumentParser

import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from art.attacks.evasion import ProjectedGradientDescentPyTorch as PGD
from art.estimators.classification import PyTorchClassifier
from loguru import logger
from tqdm import tqdm

from scaleadv.attacks.core import ScaleAttack
from scaleadv.datasets import get_imagenet
from scaleadv.datasets.transforms import Align
from scaleadv.defenses import POOLING_MAPS
from scaleadv.evaluate import Evaluator
from scaleadv.evaluate.utils import ImageManager, DataManager
from scaleadv.models import ScalingLayer
from scaleadv.models.resnet import IMAGENET_MODEL_PATH, resnet50
from scaleadv.scaling import ScalingLib, ScalingAlg, ScalingAPI
from scaleadv.utils import get_id_list_by_ratio


def run_adv(load=True):
    logger.info('Running Adversarial attack.')
    classifier = PyTorchClassifier(class_network, nn.CrossEntropyLoss(), (3,) + inp_shape, 1000, clip_values=(0, 1))
    max_iter = 30

    for eps in eps_list:
        attack = PGD(classifier, 2, eps, eps * 2.5 / max_iter, max_iter, verbose=False)
        for i in tqdm(id_list, desc=f'Adversarial Attack (eps {eps})'):
            src, y = dataset[i]
            inp = scaling_api(src[0])[None, ...]
            if load:
                adv = im.load_adv(i, eps)
                att = im.load_base(i, eps)
            else:
                adv = attack.generate(inp, np.array([y]))
                att = ScaleAttack.baseline(src, inp, adv)
                im.save_adv(i, eps, adv, att)

            data = e.eval(src, adv, att, y, y_adv=None)
            dm.save_adv(i, eps, data)


def run_att(action: str):
    logger.info(f'Running Scaling Attack: {action}.')
    classifier = PyTorchClassifier(class_network, nn.CrossEntropyLoss(), (3,) + inp_shape, 1000, clip_values=(0, 1))
    attack = ScaleAttack(scaling_api, pooling_layer, class_network, nb_samples=20, nb_flushes=20)

    for eps in eps_list:
        for i in tqdm(id_list, desc=f'Scaling Attack {action.title()} (eps {eps})'):
            src, y_src = dataset[i]
            adv = im.load_adv(i, eps)
            y_adv = classifier.predict(adv).argmax(1).item()
            if action == 'hide':
                if y_src == y_adv:  # NOTE: hide only if the adv is successful.
                    att = None
                else:
                    att = attack.hide(src, adv, y_src, y_adv, max_iter=200, weight=2, verbose=False)
            elif action == 'generate':
                _eps = eps * args.scale  # NOTE: we use the enlarged L2 eps as the recorded eps
                _iter = 100
                _eps_step = _eps * 30 / _iter
                attack_kwargs = dict(norm=2, eps=_eps, eps_step=_eps_step, max_iter=_iter, verbose=False)
                att = attack.generate(src, y_src, PGD, attack_kwargs)
            else:
                raise NotImplementedError

            if att is not None:
                im.save_att(i, eps, args.defense, action, att)
                dm.save_att(i, eps, args.defense, action, e.eval(src, adv, att, y_src, y_adv))


if __name__ == '__main__':
    p = ArgumentParser()
    _ = p.add_argument
    # Input args
    _('action', type=str, choices=('adv', 'hide', 'generate'), help='which image to generate')
    _('--model', default='none', type=str, choices=IMAGENET_MODEL_PATH.keys(), help='use robust model, optional')
    # Scaling args
    _('--lib', default='cv', type=str, choices=ScalingLib.names(), help='scaling libraries')
    _('--alg', default='linear', type=str, choices=ScalingAlg.names(), help='scaling algorithms')
    _('--scale', default=3, type=int, help='set a fixed scale ratio, unset to use the original size')
    # Scaling attack args
    _('--defense', default='none', type=str, choices=POOLING_MAPS.keys(), help='type of defense')
    # Adversarial attack args
    _('-l', '--left', default=1, type=int)
    _('-r', '--right', default=21, type=int)
    _('-s', '--step', default=1, type=int)
    _('-g', '--gpus', default='', type=str)
    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # Load data
    transform = T.Compose([Align(224, args.scale), T.ToTensor(), lambda x: np.array(x)[None, ...]])
    dataset = get_imagenet(f'val_{args.scale}', transform)

    # Load networks (pooling, scaling, class)
    src_shape = (224 * args.scale, 224 * args.scale)
    inp_shape = (224, 224)
    scaling_api = ScalingAPI(src_shape, inp_shape, args.lib, args.alg)
    scaling_layer = ScalingLayer.from_api(scaling_api).cuda()
    pooling_layer = POOLING_MAPS[args.defense].auto(args.scale * 2 - 1, scaling_api.mask).cuda()
    class_network = resnet50(args.model, normalize=True).eval().cuda()

    # Load utils
    id_list = pickle.load(open(f'static/meta/valid_ids.model_{args.model}.scale_{args.scale}.pkl', 'rb'))
    id_list = get_id_list_by_ratio(id_list, args.scale)
    eps_list = list(range(args.left, args.right, args.step))
    im = ImageManager(scaling_api)
    dm = DataManager(scaling_api)
    e = Evaluator(scaling_api, class_network, nb_samples=40)

    if args.action == 'adv':
        run_adv(load=False)
    else:
        run_att(args.action)
