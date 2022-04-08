import os
import pickle
from argparse import ArgumentParser

import torch.nn as nn
from art.estimators.classification import PyTorchClassifier
from facenet_pytorch.models.mtcnn import MTCNN
from loguru import logger

from exp.utils import savefig
from scaleadv.attacks.hsj import MyHopSkipJump
from scaleadv.attacks.sign_opt_new import SignOPT
from scaleadv.datasets.celeba import get_celeba
from scaleadv.defenses import POOLING_MAPS
from scaleadv.models.celeba import celeba_resnet34
from scaleadv.scaling import ScalingAPI, str_to_alg, str_to_lib


def attack_one(id, setid=False):
    # Load data
    x_raw, y_src = dataset[id] if setid else dataset[id_list[id]]
    x_large = mtcnn(x_raw).numpy()
    assert x_large is not None
    x_large /= 255
    logger.info(f'Loading source image: id {id}, label {y_src}, shape {x_large.shape}, dtype {x_large.dtype}.')

    # Load scaling & small image (which we want to attack)
    scaling_api = ScalingAPI(x_large.shape[-2:], (224, 224), args.lib, args.alg)
    x_small = scaling_api(x_large)[None]

    # Load classifier & attacker
    classifier = PyTorchClassifier(
        model=class_network,
        loss=nn.CrossEntropyLoss(),
        input_shape=x_small.shape[1:],
        nb_classes=2,
        clip_values=(0, 1),
        # preprocessing_defences=SaveAndLoadPyTorch()
    )

    if args.attack == 'hsj':
        attack = MyHopSkipJump(
            classifier,
            max_iter=100,
            max_eval=200,
            max_query=args.query,
            preprocess=None,
            tag=pref,
            smart_noise=False,
        )

        # Run attack (images dumped in attack)
        attack.generate(x_small, y_src[None])
    elif args.attack == 'opt':
        attack = SignOPT(classifier, k=200, preprocess=None, smart_noise=False)
        attack.generate(x_small, y_src, alpha=0.2, beta=0.001, iterations=1000, query_limit=args.query, tag=pref)
    else:
        raise NotImplementedError(args.attack)

    # Dump
    savefig(x_large, f'{pref}.src_large.png')
    savefig(x_small, f'{pref}.src_small.png')
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
    _('--scale', default=None, type=int, help='set a fixed scale ratio, unset to use the original size')
    # Scaling attack args
    _('--defense', default='none', type=str, choices=POOLING_MAPS.keys(), help='type of defense')
    _('--query', default=25000, type=int, help='query limit')
    _('--tag', default='test', type=str)
    _('--no-smart-noise', action='store_true')
    _('--no-smart-median', action='store_true')
    # Black-box attack args
    _('--attack', default='hsj', choices=['hsj', 'opt'], help='blackbox attack')
    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.g}'

    # Check test mode
    INSTANCE_TEST = args.id != -1

    # Load data
    dataset = get_celeba(attrs=['Mouth_Slightly_Open'])
    id_list = pickle.load(open('static/celeba_valid_ids/Mouth_Slightly_Open.pkl', 'rb'))[::50]

    root = f'static/celeba_bb_{args.tag}'
    os.makedirs(root, exist_ok=True)

    # Load face extractor
    mtcnn = MTCNN(image_size=224 * args.scale, post_process=False)

    # Load network
    class_network = celeba_resnet34(num_classes=11, binary_label=6, ckpt='static/models/celeba-res34.pth')
    class_network = class_network.eval().cuda()

    # attack each one
    if INSTANCE_TEST:
        pref = f'bb_{args.tag}.{args.id}.{args.defense}'
        attack_one(args.id, setid=False)
        exit()

    for i in range(args.l, args.r, args.s):
        pref = f'{root}/{i}.ratio_{args.scale}.def_{args.defense}'
        attack_one(i, setid=False)
