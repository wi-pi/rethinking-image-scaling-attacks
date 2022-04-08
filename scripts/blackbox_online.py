import os
import pickle
from argparse import ArgumentParser

import numpy as np
import torchvision.transforms as T
from art.estimators.classification import BlackBoxClassifier
from loguru import logger

from exp.utils import savefig
from scaleadv.attacks.hsj import MyHopSkipJump
from scaleadv.datasets import get_imagenet
from scaleadv.datasets.transforms import Align
from scaleadv.models import ScalingLayer
from scaleadv.models.api import OnlineModel
from scaleadv.scaling import ScalingAPI, str_to_alg, str_to_lib


def attack_one(id, setid=False):
    x_large, y_src = dataset[id] if setid else dataset[id_list[id]]
    logger.info(f'Loading source image: id {id}, label {y_src}, shape {x_large.shape}, dtype {x_large.dtype}.')

    # Load scaling
    scaling_api = ScalingAPI(x_large.shape[-2:], (224, 224), args.lib, args.alg)

    if args.scale == 1:
        x_test = scaling_api(x_large[0])[None]
        preprocess = None
    else:
        x_test = x_large
        scaling_layer = ScalingLayer.from_api(scaling_api).eval().cuda()
        preprocess = [scaling_layer, scaling_layer]

    savefig(x_large, f'{pref}.large.png')
    savefig(x_test, f'{pref}.test.png')

    # Load ART proxy
    logger.info(f'Loading test image: id {id}, label {y_src}, shape {x_test.shape}, dtype {x_test.dtype}.')
    try:
        if blackbox_model.set_current_sample(x_test) is None:
            return
    except Exception as e:
        print(e)
        return

    classifier = BlackBoxClassifier(
        blackbox_model.predict,
        input_shape=x_test.shape[1:],
        nb_classes=2,
        clip_values=(0, 1),
    )

    attack = MyHopSkipJump(classifier, max_iter=150, max_eval=200, max_query=args.query, preprocess=preprocess,
                           tag=pref)
    try:
        attack.generate(x_test, np.array([1]))  # pos label is 1
    except Exception as e:
        print(e)

    pickle.dump(attack.log, open(f'{pref}.log', 'wb'))


if __name__ == '__main__':
    p = ArgumentParser()
    _ = p.add_argument
    # Input args
    _('--id', default=-1, type=int, help='set a particular id')
    _('-l', type=int)
    _('-r', type=int)
    _('-s', type=int, default=1)
    _('-g', type=int, default=0)
    # Scaling args
    _('--lib', default='cv', type=str, choices=str_to_lib.keys(), help='scaling libraries')
    _('--alg', default='linear', type=str, choices=str_to_alg.keys(), help='scaling algorithms')
    _('--scale', default=1, type=int, help='set a fixed scale ratio, unset to use the original size')
    # Scaling attack args
    _('--query', default=25000, type=int, help='query limit')
    _('--tag', default='test', type=str)
    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.g}'

    # Check test mode
    INSTANCE_TEST = args.id != -1

    # Load data (as 3x)
    transform = T.Compose([Align(224, 3), T.ToTensor(), lambda x: np.array(x)[None, ...]])
    dataset = get_imagenet('val' if INSTANCE_TEST else f'val_3', transform)
    id_list = pickle.load(open(f'static/valid_ids.model_none.scale_3.pkl', 'rb'))[::4]

    root = f'static/online_bb_{args.tag}'
    os.makedirs(root, exist_ok=True)

    # Load network
    blackbox_model = OnlineModel()

    # attack each one
    if INSTANCE_TEST:
        pref = f'bb_{args.tag}.{args.id}.none'
        attack_one(args.id, setid=False)
        exit()

    for i in range(args.l, args.r, args.s):
        pref = f'{root}/{i}.ratio_{args.scale}.def_none'
        attack_one(i, setid=False)
