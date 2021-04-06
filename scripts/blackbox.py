import pickle
from argparse import ArgumentParser
import os

import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from art.estimators.classification import PyTorchClassifier
from loguru import logger

from scaleadv.attacks.hsj import MyHopSkipJump
from scaleadv.datasets import get_imagenet
from scaleadv.datasets.transforms import Align
from scaleadv.defenses import POOLING_MAPS
from scaleadv.models import resnet50, ScalingLayer
from scaleadv.models.resnet import IMAGENET_MODEL_PATH
from scaleadv.scaling import ScalingAPI, ScalingLib, ScalingAlg


def attack_one(id):
    src, y_src = dataset[id_list[id]]
    logger.info(f'Loading source image: id {id}, label {y_src}, shape {src.shape}, dtype {src.dtype}.')
    pref = f'static/bb/{id}.ratio_{args.scale}.def_{args.defense}'

    # Load scaling
    scaling_api = ScalingAPI(src.shape[-2:], (224, 224), args.lib, args.alg)
    scaling_layer = ScalingLayer.from_api(scaling_api).eval().cuda()

    # Load pooling
    cls = POOLING_MAPS[args.defense]
    pooling_layer = cls.auto(round(scaling_api.ratio) * 2 - 1, scaling_api.mask).eval().cuda()

    # Load network
    big_class_network = nn.Sequential(scaling_layer, class_network).eval().cuda()
    def_class_network = nn.Sequential(pooling_layer, big_class_network).eval().cuda()

    """Attack on full
    Problem: now we only have a linear approximation of median filter.
    """
    classifier = PyTorchClassifier(def_class_network, nn.CrossEntropyLoss(), src.shape[1:], 1000, clip_values=(0, 1))
    preprocess = [scaling_layer, scaling_layer]
    if args.defense != 'none':
        preprocess = [nn.Sequential(POOLING_MAPS['quantile'].like(pooling_layer), scaling_layer),
                      nn.Sequential(pooling_layer, scaling_layer)]
    attack = MyHopSkipJump(classifier, max_iter=100, max_eval=600, max_query=args.query, preprocess=preprocess,
                           tag=pref)
    attack.generate(src)
    pickle.dump(attack.log, open(f'{pref}.log', 'wb'))


if __name__ == '__main__':
    p = ArgumentParser()
    _ = p.add_argument
    # Input args
    _('--model', default='none', type=str, choices=IMAGENET_MODEL_PATH.keys(), help='use robust model, optional')
    _('-l', type=int)
    _('-r', type=int)
    _('-s', type=int)
    _('-g', type=int, default=0)
    # Scaling args
    _('--lib', default='cv', type=str, choices=ScalingLib.names(), help='scaling libraries')
    _('--alg', default='linear', type=str, choices=ScalingAlg.names(), help='scaling algorithms')
    _('--scale', default=None, type=int, help='set a fixed scale ratio, unset to use the original size')
    # Scaling attack args
    _('--defense', default='none', type=str, choices=POOLING_MAPS.keys(), help='type of defense')
    _('--query', default=25000, type=int, help='query limit')
    _('--tag', default='test', type=str)
    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.g}'

    # Load data
    transform = T.Compose([Align(224, args.scale), T.ToTensor(), lambda x: np.array(x)[None, ...]])
    dataset = get_imagenet(f'val_3', transform)
    id_list = pickle.load(open(f'static/meta/valid_ids.model_{args.model}.scale_3.pkl', 'rb'))[::4]

    os.makedirs('static/bb', exist_ok=True)

    # Load network
    class_network = resnet50(robust=args.model, normalize=True).eval().cuda()

    # attack each one
    for i in range(args.l, args.r, args.s):
        attack_one(i)

    """Attack on filtered HR
    """
    # classifier = PyTorchClassifier(big_class_network, nn.CrossEntropyLoss(), (3, 672, 672), 1000, clip_values=(0, 1))
    # attack = MyHopSkipJump(classifier, max_query=args.query, preprocess=scaling_layer, tag=args.tag)
    # src_med = pooling_layer(torch.as_tensor(src).cuda()).cpu().numpy()
    # # adv = attack.generate(src_med)
    # adv = F.to_tensor(Image.open('bb_test.15.png'))[None, ...].numpy()
    # pooling_layer = POOLING_MAPS['quantile'].like(pooling_layer)
    # x = np.clip(src_med + (adv - src_med) * 1.9, 0, 1)
    # att = inverse_median(med_class_network, pooling_layer, src, x, w1=1, w2=2, T=1000, tau=50, lr=0.01)

    """Attack on filtered LR
    """
    # classifier = PyTorchClassifier(class_network, nn.CrossEntropyLoss(), (3, 224, 224), 1000, clip_values=(0, 1))
    # attack = MyHopSkipJump(classifier, max_query=args.query, preprocess=None, tag=args.tag)
    # inp = scaling_layer(pooling_layer(torch.as_tensor(src).cuda())).cpu().numpy()
    # adv = attack.generate(inp)

    """Test SimBA
    Problem: unsure how to get orthonormal basis in the before-median space.
    """
    # classifier = PyTorchClassifier(class_network, nn.CrossEntropyLoss(), (3, 224, 224), 1000, clip_values=(0, 1))
    # attack = MySimBA(classifier, 'px', max_iter=10000, epsilon=0.2)
    # adv = attack.generate(src)
    # F.to_pil_image(torch.tensor(adv[0])).save(f'test.png')
