import os

import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from art.attacks.evasion import ProjectedGradientDescentPyTorch as PGD
from art.estimators.classification import PyTorchClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm

from scaleadv.datasets.imagenet import IMAGENET_MEAN, IMAGENET_STD
from scaleadv.datasets.utils import ImageFilesDataset
from scaleadv.models.layers import NormalizationLayer
from scaleadv.tests.gen_adv_pgd import get_model

OUTPUT_PATH = 'static/results/imagenet-600'
MODEL_PATH = {
    np.inf: 'static/models/imagenet_linf_4.pt',
    2: 'static/models/imagenet_l2_3_0.pt',
}
NORM = np.inf
NORM_STR = 'inf'
EPSILON = list(range(17))
PGD_ITER = 20
EPSILON_DIV = 255. if NORM == np.inf else 1.


def gen_dataset(loader, attacker, eps):
    for names, x_batch in tqdm(loader, desc=f'NORM = {NORM_STR}, EPS = {eps}'):
        x_adv = attacker.generate(x_batch)
        for n, o in zip(names, x_adv):
            o = (o * 255).round().astype(np.uint8).transpose(1, 2, 0)
            n = os.path.join(OUTPUT_PATH, n.replace('.png', f'.adv_L{NORM_STR}_{eps}.png'))
            Image.fromarray(o).save(n)


if __name__ == '__main__':
    # load data
    ds1 = ImageFilesDataset(root=OUTPUT_PATH, suffix='.src_inp.png', transform=T.ToTensor())
    ds2 = ImageFilesDataset(root=OUTPUT_PATH, suffix='.src_def_inp.png', transform=T.ToTensor())
    ld1 = DataLoader(ds1, batch_size=256, num_workers=8)
    ld2 = DataLoader(ds2, batch_size=256, num_workers=8)

    # load classifier
    model = nn.Sequential(
        NormalizationLayer(IMAGENET_MEAN, IMAGENET_STD),
        get_model(weights_file=MODEL_PATH[NORM]),
    )
    model = nn.DataParallel(model).eval().cuda()
    classifier = PyTorchClassifier(model, nn.CrossEntropyLoss(), (3, 224, 224), 1000, clip_values=(0., 1.))

    # run for each epsilon
    for eps in EPSILON:
        sigma = eps * 2.5 / PGD_ITER
        attacker = PGD(classifier, NORM, eps / EPSILON_DIV + 1e-8, sigma / EPSILON_DIV + 1e-8, max_iter=PGD_ITER)
        gen_dataset(ld1, attacker, eps)
        gen_dataset(ld2, attacker, eps)
