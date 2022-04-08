import pickle
from argparse import ArgumentParser
import os

import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from art.estimators.classification import PyTorchClassifier
from facenet_pytorch.models.mtcnn import MTCNN
from loguru import logger

from scaleadv.attacks.hsj import MyHopSkipJump
from scaleadv.attacks.sign_opt_new import SignOPT
from scaleadv.datasets import get_imagenet
from scaleadv.datasets.celeba import get_celeba
from scaleadv.datasets.transforms import Align
from scaleadv.defenses import POOLING_MAPS
from scaleadv.models import resnet50, ScalingLayer
from scaleadv.models.celeba import celeba_resnet34
from scaleadv.models.resnet import IMAGENET_MODEL_PATH
from scaleadv.scaling import ScalingAPI, ScalingLib, ScalingAlg, str_to_alg, str_to_lib


def attack_one(id, setid=False):
    # Load data
    x_raw, y_src = dataset[id] if setid else dataset[id_list[id]]
    src = mtcnn(x_raw).numpy()[None]
    assert src is not None
    src /= 255
    logger.info(f'Loading source image: id {id}, label {y_src}, shape {src.shape}, dtype {src.dtype}.')

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
    classifier = PyTorchClassifier(def_class_network, nn.CrossEntropyLoss(), src.shape[1:], 2, clip_values=(0, 1))
    preprocess = [scaling_layer, scaling_layer]
    if args.defense != 'none':
        if args.no_smart_median:
            preprocess = [nn.Sequential(pooling_layer, scaling_layer),
                          nn.Sequential(pooling_layer, scaling_layer)]
        else:
            preprocess = [nn.Sequential(POOLING_MAPS['quantile'].like(pooling_layer), scaling_layer),
                          nn.Sequential(pooling_layer, scaling_layer)]

    if args.attack == 'hsj':
        attack = MyHopSkipJump(classifier, max_iter=150, max_eval=200, max_query=args.query, preprocess=preprocess,
                               tag=pref, smart_noise=not args.no_smart_noise)
        attack.generate(src, y_src[None])
    elif args.attack == 'opt':
        attack = SignOPT(classifier, k=200, preprocess=preprocess, smart_noise=not args.no_smart_noise)
        attack.generate(src, y_src, alpha=0.2, beta=0.001, iterations=1000, query_limit=args.query, tag=pref)
    else:
        raise NotImplementedError(args.attack)

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
