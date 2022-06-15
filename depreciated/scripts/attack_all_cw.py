import os
import pickle
from argparse import ArgumentParser
from operator import itemgetter

import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from art.attacks.evasion import CarliniL2Method
from art.estimators.classification import PyTorchClassifier
from loguru import logger
from tqdm import tqdm

from depreciated.scaleadv import ScaleAttack
from depreciated.scaleadv import get_imagenet
from depreciated.scaleadv import Align
from depreciated.scaleadv import POOLING_MAPS
from depreciated.scaleadv.evaluate import Evaluator
from depreciated.scaleadv.evaluate.utils import ImageManager, DataManager
from depreciated.scaleadv import ScalingLayer
from depreciated.scaleadv import IMAGENET_MODEL_PATH, resnet50
from depreciated.scaleadv import ScalingLib, ScalingAlg, ScalingAPI
from depreciated.scaleadv.utils import get_id_list_by_ratio


def run_adv(load=True):
    logger.info('Running Adversarial attack.')
    classifier = PyTorchClassifier(class_network, nn.CrossEntropyLoss(), (3,) + inp_shape, 1000, clip_values=(0, 1))

    for eps in eps_list:
        attack = CarliniL2Method(classifier, confidence=eps, binary_search_steps=20, max_iter=100, verbose=False)
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
            adv = im_ADV.load_adv(i, eps)
            y_adv = classifier.predict(adv).argmax(1).item()
            if action == 'hide':
                if y_src == y_adv:  # NOTE: hide only if the adv is successful.
                    att = None
                else:
                    att = attack.hide(src, adv, y_src, y_adv, max_iter=200, weight=2, verbose=False)
            elif action == 'generate':
                attack_kwargs = dict(confidence=eps, binary_search_steps=20, max_iter=100, verbose=False)
                att = attack.generate(src, y_src, CarliniL2Method, attack_kwargs)
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
    _('-l', '--left', default=0, type=int)
    _('-r', '--right', default=100, type=int)
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
    id_list = get_id_list_by_ratio(id_list, args.scale)[::2]
    id_list = list(itemgetter(*range(args.left, args.right, args.step))(id_list))
    eps_list = list(range(11))  # list(range(args.left, args.right, args.step))
    im_ADV = ImageManager(scaling_api)
    im = ImageManager(scaling_api, tag='.cw_med_it100')
    dm = DataManager(scaling_api, tag='.cw_med_it100')
    e = Evaluator(scaling_api, class_network, nb_samples=1)

    if args.action == 'adv':
        run_adv(load=False)
    else:
        run_att(args.action)
