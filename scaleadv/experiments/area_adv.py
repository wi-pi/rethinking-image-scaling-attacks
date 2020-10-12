from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from itertools import product

import cvxpy as cp
import numpy as np
import numpy.linalg as LA
import torch.nn as nn
from PIL import Image
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier
from attack.area_attack.AreaNormEnumType import AreaNormEnumType
from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries
from torchvision.models import resnet50
from tqdm import tqdm

from scaleadv.bypass.random import resize_to_224x
from scaleadv.datasets.imagenet import IMAGENET_MEAN, IMAGENET_STD
from scaleadv.datasets.imagenet import create_dataset
from scaleadv.models.layers import NormalizationLayer


def solve(src: np.ndarray, tgt: int, eps: int, norm: AreaNormEnumType):
    block_shape = src.shape
    src = src.flatten()
    res = cp.Variable(shape=src.shape)
    delta = res - src
    if norm is AreaNormEnumType.L2:
        obj = 0.5 * cp.sum_squares(delta)  # quad_form(delta, np.identity(src.shape))
    else:
        obj = cp.sum(cp.abs(delta))

    prob = cp.Problem(cp.Minimize(obj), [
        cp.abs(cp.sum(res) - tgt * len(src)) <= eps / 2,
        res >= 0,
        res <= 255
    ])
    try:
        prob.solve()
    except:
        prob.solve(solver=cp.ECOS)

    return res.value.round().reshape(block_shape)


def area_attack_one_dim_parallel(src: np.ndarray, tgt: np.ndarray, kx: int, ky: int, eps: int,
                                 norm: AreaNormEnumType, verbose: bool = False):
    assert src.ndim == 2 and tgt.ndim == 2
    result = np.zeros_like(src)

    with ProcessPoolExecutor() as exe:
        future_to_block = {}
        # generate cp.Prob for each block
        for x, y in product(*map(range, tgt.shape)):
            # locate block
            block_x = slice(x * kx, x * kx + kx)
            block_y = slice(y * ky, y * ky + ky)
            s = src[block_x, block_y]
            t = tgt[x, y]

            # generate problem
            future_to_block[exe.submit(solve, s, t, eps, norm)] = x, y, block_x, block_y

        # aggregate results
        for future in tqdm(as_completed(future_to_block), total=len(future_to_block)):
            x, y, block_x, block_y = future_to_block[future]
            try:
                block = future.result()
            except:
                print('FAIL:', src[block_x, block_y].flatten(), tgt[x, y])
                block = src[block_x, block_y]

            result[block_x, block_y] = block

    return result


def area_attack(src: np.ndarray, tgt: np.ndarray, norm: AreaNormEnumType, eps: int):
    assert src.ndim == 3 and tgt.ndim == 3
    assert all(src.shape[i] % tgt.shape[i] == 0 for i in [0, 1])
    kx, ky = [src.shape[i] // tgt.shape[i] for i in [0, 1]]
    res = np.zeros_like(src)
    for c in range(src.shape[2]):
        res[..., c] = area_attack_one_dim_parallel(src[..., c], tgt[..., c], kx, ky, eps, norm)
    return res


if __name__ == '__main__':
    # params
    ID = 5000
    NORM = 2
    SIGMA = 10
    STEP = 30
    EPSILON = SIGMA * 2.5 / STEP
    LIB = SuppScalingLibraries.CV
    ALGO = SuppScalingAlgorithms.AREA

    # load data
    dataset = create_dataset(transform=None)
    _, x_img, y_img = dataset[ID]
    x_img = resize_to_224x(x_img)
    x_src = np.array(x_img)

    # load art proxy
    model = nn.Sequential(NormalizationLayer(IMAGENET_MEAN, IMAGENET_STD), resnet50(pretrained=True)).eval()
    classifier = PyTorchClassifier(model, nn.CrossEntropyLoss(), (3, 224, 224), 1000, clip_values=(0, 1))
    adv_attack = ProjectedGradientDescentPyTorch(classifier, NORM, SIGMA, EPSILON, STEP)

    # load scaling attack
    scaling = ScalingGenerator.create_scaling_approach(x_src.shape, (224, 224, 3), LIB, ALGO)

    # get adv-example on area-filtered inp
    x_inp = scaling.scale_image(x_src)
    x_inp_t = np.array(x_inp / 255., dtype=np.float32).transpose((2, 0, 1))[None, ...]
    x_adv_t = adv_attack.generate(x_inp_t)

    # TEST: benign & adv
    y_inp = classifier.predict(x_inp_t).argmax(1)
    print('y_inp', y_inp)
    y_adv = classifier.predict(x_adv_t).argmax(1)
    print('y_adv', y_adv)

    # hide x_adv in x_src
    x_adv = np.array(x_adv_t[0] * 255).transpose((1, 2, 0)).round().astype(np.uint8)
    x_scl = area_attack(x_src, x_adv, AreaNormEnumType.L1, eps=1)

    # TEST: scl
    x_scl_inp = scaling.scale_image(x_scl)
    x_scl_inp_t = np.array(x_scl_inp / 255., dtype=np.float32).transpose((2, 0, 1))[None, ...]
    y_scl = classifier.predict(x_scl_inp_t).argmax(1)
    print('y_scl', y_scl)

    # save
    cap = ['src', 'inp', 'adv', 'scl', 'scl_inp']
    for c in cap:
        var = locals()[f'x_{c}']
        Image.fromarray(var).save(f'area-{ID}-{c}.png')

    # compare L2
    x_scl_inp = scaling.scale_image(x_scl)
    print('inp - adv', LA.norm((0.0 + x_inp - x_adv).flatten() / 255, ord=2))
    print('src - scl', LA.norm((0.0 + x_src - x_scl).flatten() / 255, ord=2))
    print('scl - adv', LA.norm((0.0 + x_scl_inp - x_adv).flatten() / 255, ord=2))
