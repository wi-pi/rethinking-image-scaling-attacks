import os
from collections import OrderedDict

import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import transforms as T
from PIL import Image
from art.attacks.evasion import ProjectedGradientDescentPyTorch

from scaleadv.datasets.imagenet import create_dataset
from scaleadv.attacks.adv import AdvAttack
from scaleadv.attacks.scale import ScaleAttack, SuppScalingLibraries, SuppScalingAlgorithms
from scaling.ScalingGenerator import ScalingGenerator
from scaleadv.tests.models import get_classifier
from scaleadv.tests.scale import save


def get_adv(img, attack, crop=False):
    # preprocess
    if crop:
        img = F.resize(img, size=(256, 256), interpolation=Image.NEAREST)
        img = F.center_crop(img, output_size=(224, 224))
    else:
        img = F.resize(img, size=(224, 224), interpolation=Image.NEAREST)
    x = np.array(img)

    # from np.ndarray[0,255][WHC] to torch.tensor[0,1][BCWH]
    x = x.transpose(2, 0, 1)
    x = torch.as_tensor(x / 255, dtype=torch.float32)
    x = x[None, ...]

    # adv attack
    x = attack.generate(x)

    # from torch.tensor[0,1][BCWH] to np.ndarray[0,255][WHC]
    x = x[0]
    x = np.array(x * 255)
    x = x.transpose(1, 2, 0)

    return x

def get_scl(img, x_adv, kwargs, crop=False):
    # preprocess src
    src = np.array(img, dtype=np.uint8)

    # preprocess tgt
    if crop:
        tgt = F.resize(img, size=(256, 256), interpolation=Image.NEAREST)
        tgt = np.array(tgt)
        
        # put x_adv at the center of tgt
        sw, sh, _ = tgt.shape
        tw, th, _ = x_adv.shape
        sx = int(round((sw - tw) / 2.))
        sy = int(round((sh - th) / 2.))
        tgt[sx:sx+tw, sy:sy+th] = x_adv
        assert sx == sy == 16
    else:
        tgt = np.array(x_adv, dtype=np.uint8)

    # scale attck
    attack = ScaleAttack(**kwargs)
    imgs = attack.generate(src, tgt)

    # down scale
    scale = ScalingGenerator.create_scaling_approach(src.shape, tgt.shape, lib, algo)

    return src, tgt, imgs, scale

def predict(classifier, fname, path=''):
    # load image
    fname = os.path.join(path, fname)
    img = Image.open(fname)
    
    # preprocess
    trans = T.Compose([
        T.Resize((224, 224), Image.NEAREST),
        T.ToTensor(),
        lambda x: x[None, ...]
    ])
    x = trans(img)

    # predict
    y = classifier.predict(x).argmax(1).item()
    return y


if __name__ == '__main__':
    # get data
    test_id = 10000
    path = 'static/results/scaleadv'
    os.makedirs(path, exist_ok=True)

    # load data
    data = create_dataset('static/datasets/imagenet/val/', transform=None)
    img, y = data[test_id]

    # get adv attacker
    classifier = get_classifier()
    art = ProjectedGradientDescentPyTorch(classifier, norm=np.inf, eps=64/255., eps_step=4/255., max_iter=20)
    adv_attack = AdvAttack(art)

    # get adv
    x_adv = get_adv(img, adv_attack)

    # get scl attacker
    lib = SuppScalingLibraries.PIL
    algo = SuppScalingAlgorithms.NEAREST
    kwargs = {
        'lib': lib,
        'algo': algo,
        'eps': 1,
        'bandwidth': 2,
        'allowed_changes': 0.75,
        'verbose': False,
    }

    # get scl
    src, tgt, imgs, scale = get_scl(img, x_adv, kwargs)

    # save figs
    caps = 'x_scl', 'x_scl_defense', 'x_ada', 'x_ada_defense'
    for xb, c in zip(imgs, caps):
        xs = scale.scale_image(xb)
        save(xb, f'{test_id}.{c}.big', path)
        save(xs, f'{test_id}.{c}.small', path)
    save(src, f'{test_id}.src', path)
    save(tgt, f'{test_id}.tgt', path)
    
    # check predictions
    preds = OrderedDict({'y': y})
    preds['raw'] = predict(classifier, f'{test_id}.src.png', path)
    preds['adv'] = predict(classifier, f'{test_id}.tgt.png', path)
    for tag in caps:
        preds[f'{tag}.big'] = predict(classifier, f'{test_id}.{tag}.big.png', path)
        preds[f'{tag}.small'] = predict(classifier, f'{test_id}.{tag}.small.png', path)
    for tag, y in preds.items():
        print(f'{tag:20s} {y}')

