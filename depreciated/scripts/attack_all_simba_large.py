import argparse
import os
import pickle

import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from art.estimators.classification import PyTorchClassifier
from loguru import logger

from depreciated.exp.utils import savefig
from depreciated.scaleadv import MySimBA
from depreciated.scaleadv import get_imagenet
from depreciated.scaleadv import Align
from depreciated.scaleadv import SaveAndLoadPyTorch
from depreciated.scaleadv import ScalingLayer
from depreciated.scaleadv import IMAGENET_MODEL_PATH, resnet50


def attack_one(id, setid=False):
    x_large, y_src = dataset[id] if setid else dataset[id_list[id]]
    logger.info(f'Loading source image: id {id}, label {y_src}, shape {x_large.shape}, dtype {x_large.dtype}.')

    # Load scaling api & layer & downscaled source image
    scaling_api = ScalingAPI(x_large.shape[-2:], (224, 224), args.lib, args.alg)
    scaling_layer = ScalingLayer.from_api(scaling_api).eval().cuda()
    x_small = scaling_api(x_large[0])[None]

    # Load networks
    large_network = nn.Sequential(scaling_layer, small_network).eval().cuda()

    # Load classifiers
    classifier_small = PyTorchClassifier(small_network, nn.CrossEntropyLoss(), x_small.shape[1:], 1000, clip_values=(0, 1), preprocessing_defences=SaveAndLoadPyTorch())
    classifier_large = PyTorchClassifier(large_network, nn.CrossEntropyLoss(), x_large.shape[1:], 1000, clip_values=(0, 1), preprocessing_defences=SaveAndLoadPyTorch())
    logger.info(f'Predicted x_small as {classifier_small.predict(x_small).argmax(1)}.')
    logger.info(f'Predicted x_large as {classifier_large.predict(x_large).argmax(1)}.')

    # Attack
    attack = MySimBA(classifier_large, 'dct', max_iter=args.query, epsilon=0.2)
    x_large_adv = attack.generate(x_large)
    logger.info(f'Predicted x_large_adv as {classifier_large.predict(x_large_adv).argmax(1)}.')

    # pickle.dump(attack.log, open(f'{pref}.log', 'wb'))
    savefig(x_large_adv, f'{pref}.large.png')

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
