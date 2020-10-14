import concurrent.futures
from math import ceil
from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from art.estimators.classification import PyTorchClassifier
from attack.QuadrScaleAttack import QuadraticScaleAttack
from attack.adaptive_attack.AdaptiveAttackPreventionGenerator import AdaptiveAttackPreventionGenerator
from defenses.detection.fourier.FourierPeakMatrixCollector import FourierPeakMatrixCollector, PeakMatrixMethod
from defenses.prevention.PreventionDefenseGenerator import PreventionDefenseGenerator
from defenses.prevention.PreventionDefenseType import PreventionTypeDefense
from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries
from torch.nn import DataParallel
from torchvision.models import resnet50
from tqdm import tqdm

from scaleadv.attacks.shadow import CustomShadowAttack, SmoothAttack
from scaleadv.datasets.imagenet import IMAGENET_MEAN, IMAGENET_STD
from scaleadv.datasets.imagenet import create_dataset
from scaleadv.models.layers import NormalizationLayer

ID = 30000
LIB = SuppScalingLibraries.CV
ALGO = SuppScalingAlgorithms.LINEAR
STEP = 300
BATCH = 400
MAX_BATCH = 400
OUTPUT = f'static/results/bypass/random'


def protect(x):
    return scaling.scale_image(defense.make_image_secure(x))  # Note: cython module fixed seed, don't forgot.


def get_protect(x, n=20, use_cache=False, save=False):
    assert (not use_cache) or (use_cache and not save)
    if use_cache:
        cache = np.load(f'noise-{ID}-{MAX_BATCH}.npy')
        idx = np.random.choice(len(cache), n)
        return cache[idx]

    with concurrent.futures.ProcessPoolExecutor() as exec:
        output = list(tqdm(exec.map(protect, [x] * n), total=n))
    output = np.stack(output, axis=0)
    if save:
        np.save(f'noise-{ID}-{BATCH}.npy', output)
    return output


class PrintableCrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self, *args, **kwargs):
        super(PrintableCrossEntropyLoss, self).__init__(*args, **kwargs)

    def forward(self, input, target):
        loss = super(PrintableCrossEntropyLoss, self).forward(input, target)
        print(f'{loss.cpu().item():.3f}', end='\t')
        return loss


def topk_acc(pred: np.ndarray, y: int, k: int = 1):
    topk = pred.argsort()[:, -k:]
    good = np.equal(topk, y).any(axis=1).astype(np.float32)
    return good.mean()


def show(caption: str, pred: np.ndarray, y: int, k: List[int]):
    acc = [topk_acc(pred, y, i) for i in k]
    print(caption, ' '.join([f'{a:.2%}' for a in acc]))


def resize_to_224x(img, more=1):
    w, h = img.size
    w = 224 * ceil(w / 224) * more
    h = 224 * ceil(h / 224) * more
    return img.resize((w, h))


if __name__ == '__main__':
    # load data
    dataset = create_dataset(transform=None)
    _, x_img, y_img = dataset[ID]
    x_img = resize_to_224x(x_img)
    x_src = np.array(x_img)

    # load classifier
    net = resnet50  # resnet50
    model = nn.Sequential(NormalizationLayer(IMAGENET_MEAN, IMAGENET_STD), net(pretrained=True)).eval()
    model = DataParallel(model).cuda()
    classifier = PyTorchClassifier(model, PrintableCrossEntropyLoss(), (3, 224, 224), 1000, clip_values=(0., 1.))

    # load adv attack
    kwargs = {}  # {'lambda_tv': 0.03, 'lambda_c': 0.1, 'lambda_s': 0.05}
    adv_attack = CustomShadowAttack(classifier, **kwargs, nb_steps=STEP, batch_size=BATCH, targeted=False,
                                    learning_rate=0.3)
    off_attack = SmoothAttack(model)

    # load scaling attack
    fpm = FourierPeakMatrixCollector(PeakMatrixMethod.optimization, ALGO, LIB)
    scaling = ScalingGenerator.create_scaling_approach(x_src.shape, (224, 224, 3), LIB, ALGO)
    scl_attack = QuadraticScaleAttack(eps=1.0, verbose=False)
    defense = PreventionDefenseGenerator.create_prevention_defense(
        defense_type=PreventionTypeDefense.randomfiltering,
        scaler_approach=scaling,
        fourierpeakmatrixcollector=fpm,
        bandwidth=2,
        verbose_flag=False,
        usecythonifavailable=True
    )
    adaptive_attack = AdaptiveAttackPreventionGenerator.create_adaptive_attack(
        defense_type=PreventionTypeDefense.randomfiltering,
        scaler_approach=scaling,
        preventiondefense=defense,
        verbose_flag=False,
        usecythonifavailable=True,
        choose_only_unused_pixels_in_overlapping_case=False,
        allowed_changes=0.8
    )

    # warm start & get noisy inp
    defense.make_image_secure(x_src)
    noise = get_protect(x_src, n=BATCH, use_cache=False, save=True)
    noise = noise.transpose((0, 3, 1, 2)) / 255.

    # run adv attack
    x_inp = scaling.scale_image(x_src)
    noise[:20] = x_inp = x_inp.transpose((2, 0, 1)) / 255.
    _, delta = off_attack.perturb_with_given_noise(x=torch.tensor(x_inp, dtype=torch.float32),
                                                   noise=torch.tensor(noise, dtype=torch.float32),
                                                   y=200, batch=BATCH, steps=STEP, print_stats=True)
    # to [CHW] ndarray in [0, 1]
    delta = np.array(delta, dtype=np.float32)


    def test(x, y, d=np.array(0.0)):
        x = x.astype(np.float32).transpose((0, 3, 1, 2)) / 255. + d
        p = np.mean(classifier.predict(x.clip(0, 1)).argmax(1) == y)
        return p


    # T(src) + delta
    pred = test(scaling.scale_image(x_src)[None, ...], 200, delta)
    print(f'T(src) + delta = {pred:.2%}')

    # T(src + e1) + delta
    pred = test(get_protect(x_src, n=100), 200, delta)
    print(f'T(src + e1) + delta = {pred:.2%}')

    # run scaling attack
    x_adv = np.clip(x_inp + delta, 0, 1)
    x_adv = (x_adv * 255).round().astype(np.uint8).transpose((1, 2, 0))
    Image.fromarray(x_adv).save(f'{ID}.x_adv.png')
    x_scl, _, _ = scl_attack.attack(x_src, x_adv, scaling)

    # T(scl)
    pred = test(scaling.scale_image(x_scl)[None, ...], 200)
    print(f'T(scl) = {pred:.2%}')

    # T(scl + e2)
    pred = test(get_protect(x_scl, n=100), 200)
    print(f'T(src + e1) = {pred:.2%}')

    # run adaptive attack
    x_ada = adaptive_attack.counter_attack(x_scl)

    # T(ada)
    pred = test(scaling.scale_image(x_ada)[None, ...], 200)
    print(f'T(ada) = {pred:.2%}')

    # T(ada + e3)
    pred = test(get_protect(x_ada, n=100), 200)
    print(f'T(ada + e3) = {pred:.2%}')

    from IPython import embed; embed(using=False); exit()

    # test results
    xs = ['x_src', 'x_scl']
    for caption in xs:
        # locate variable
        var = locals()[caption]

        # save
        Image.fromarray(var).save(f'{ID}.{caption}.png')
        # test raw image
        inp = scaling.scale_image(var).transpose((2, 0, 1)).astype(np.float32)[None, ...] / 255
        pred = classifier.predict(inp)
        show(caption + ' (raw,  1)', pred, y_img, k=[1, 5])
        # test defense image
        inp = get_protect(var, n=40)
        inp = inp.astype(np.float32).transpose((0, 3, 1, 2)) / 255.
        pred = classifier.predict(inp)
        show(caption + ' (def, 40)', pred, y_img, k=[1, 5])
