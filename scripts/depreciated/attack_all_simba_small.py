import argparse
import os
import pickle

import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from art.estimators.classification import PyTorchClassifier
from loguru import logger

from exp.utils import savefig
from scaleadv.attacks.simba import MySimBA
from scaleadv.datasets import get_imagenet
from scaleadv.datasets.transforms import Align
from scaleadv.defenses.preprocessor import SaveAndLoadPyTorch
from scaleadv.models.resnet import IMAGENET_MODEL_PATH, resnet50
from scaleadv.scaling import *


def attack_one(id, setid=False):
    src, y_src = dataset[id] if setid else dataset[id_list[id]]
    logger.info(f'Loading source image: id {id}, label {y_src}, shape {src.shape}, dtype {src.dtype}.')

    # Load scaling
    scaling_api = ScalingAPI(src.shape[-2:], (224, 224), args.lib, args.alg)
    x_small = scaling_api(src[0])[None]

    classifier = PyTorchClassifier(small_network, nn.CrossEntropyLoss(), x_small.shape[1:], 1000, clip_values=(0, 1),
                                   preprocessing_defences=SaveAndLoadPyTorch())
    logger.info(f'Predicted x_small as {classifier.predict(x_small).argmax(1)}.')

    # Attack
    attack = MySimBA(classifier, 'dct', max_iter=args.query, epsilon=0.2)
    x_small_adv = attack.generate(x_small)
    logger.info(f'Predicted x_small_adv as {classifier.predict(x_small_adv).argmax(1)}.')

    pickle.dump(attack.log, open(f'{pref}.log', 'wb'))
    savefig(x_small_adv, f'{pref}.png')

    # # reload
    # x_reload = to_tensor(Image.open(f'{pref}.png')).numpy()[None]
    # logger.info(f'Predicted reloaded adv as {classifier.predict(x_reload).argmax(1)}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _ = parser.add_argument
    # Inputs
    _('--id', default=-1, type=int, help='ID of the test image.')
    _('--model', default='none', type=str, choices=IMAGENET_MODEL_PATH.keys(), help='Optional robust model.')
    _('-l', type=int)
    _('-r', type=int)
    _('-s', type=int, default=1)
    _('-g', type=int, default=0)
    # Scaling
    _('--lib', default='cv', type=str, choices=str_to_lib.keys(), help='Scaling library.')
    _('--alg', default='linear', type=str, choices=str_to_alg.keys(), help='Scaling algorithm.')
    # Standard evasion attack args
    _('--query', default=25000, type=int, help='query limit')
    # Scaling attack args
    pass
    # Persistent
    pass
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.g}'

    # Check test mode
    INSTANCE_TEST = args.id != -1

    # Load data
    transform = T.Compose([Align(256, 3), T.ToTensor(), lambda x: np.array(x)[None, ...]])
    dataset = get_imagenet('val' if INSTANCE_TEST else 'val_3', transform)
    id_list = pickle.load(open(f'static/meta/valid_ids.model_{args.model}.scale_3.pkl', 'rb'))[::2]

    root = 'static/simba'
    os.makedirs(root, exist_ok=True)

    # Load network
    small_network = resnet50(robust=args.model, normalize=True).eval().cuda()

    # attack one
    if INSTANCE_TEST:
        pref = f'simba_test.{args.id}'
        attack_one(args.id, setid=True)
        exit()

    # attack each one
    for i in range(args.l, args.r, args.s):
        pref = f'{root}/{i}.{args.lib}.{args.alg}.ratio_1'
        attack_one(i, setid=False)
