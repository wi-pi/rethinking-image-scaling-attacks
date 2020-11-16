import argparse
import os

import numpy as np
import torch as ch
import torch.nn as nn
from PIL import Image
from art.attacks.evasion import ProjectedGradientDescentPyTorch as PGD
from art.estimators.classification import PyTorchClassifier
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet50
from tqdm import tqdm

from scaleadv.datasets.imagenet import create_dataset, IMAGENET_TRANSFORM_NOCROP, IMAGENET_MEAN, IMAGENET_STD
from scaleadv.models.layers import NormalizationLayer

DATASET_PATH = 'static/datasets/imagenet/val/'
MODEL_PATH = {
    np.inf: 'static/models/imagenet_linf_4.pt',
    2: 'static/models/imagenet_l2_3_0.pt',
}


def get_model(weights_file: str = None):
    pretrained = weights_file is None
    network = resnet50(pretrained=pretrained)

    if weights_file:
        prefix = 'module.model.'
        ckpt = ch.load(weights_file).get('model', {})
        ckpt = {k.replace(prefix, ''): v for k, v in ckpt.items() if k.startswith(prefix)}
        network.load_state_dict(ckpt)

    return network


if __name__ == '__main__':
    # load params
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--norm', type=int, default=np.inf, help='Norm of PGD attack.')
    p.add_argument('-e', '--eps', type=float, required=True, help='Epsilon.')
    p.add_argument('-s', '--sigma', type=float, required=True, help='Sigma.')
    p.add_argument('-i', '--iter', type=int, required=True, help='PGD iterations.')
    p.add_argument('-b', '--batch', type=int, default=128, help='Batch size.')
    p.add_argument('-t', '--target', type=int, help='Targeted class.', )
    p.add_argument('-o', '--output', type=str, default=None, metavar='FILE', help='Output directory.')
    p.add_argument('--skip', type=int, default=50, help='Test on every SKIP examples.')
    args = p.parse_args()
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    print(args)

    # load data
    dataset = create_dataset(root=DATASET_PATH, transform=IMAGENET_TRANSFORM_NOCROP)
    dataset = Subset(dataset, indices=range(0, len(dataset), args.skip))
    loader = DataLoader(dataset, batch_size=args.batch, num_workers=8)

    # load classifier
    model = nn.Sequential(
        NormalizationLayer(IMAGENET_MEAN, IMAGENET_STD),
        get_model(weights_file=MODEL_PATH[args.norm]),
    )
    model = nn.DataParallel(model)
    model = model.eval().cuda()
    classifier = PyTorchClassifier(model, nn.CrossEntropyLoss(), (3, 224, 224), 1000, clip_values=(0., 1.))

    # load attacker
    targeted = args.target is not None
    # TODO: support targeted attack.
    attacker = PGD(classifier, norm=args.norm, eps=args.eps, eps_step=args.sigma, max_iter=args.iter,
                   targeted=targeted, batch_size=args.batch)

    # evaluate
    y_true, y_pred, y_evil = [], [], []
    for ids, x, y in tqdm(loader):
        x_evil = attacker.generate(x, y)
        y_true.append(y)
        y_pred.append(classifier.predict(x))
        y_evil.append(classifier.predict(x_evil))

        if args.output:
            x_evil = (x_evil * 255).round().astype(np.uint8).transpose(0, 2, 3, 1)
            for i, x in zip(ids, x_evil):
                name = os.path.join(args.output, f'{i}.png')
                Image.fromarray(x).save(name)

    # save stats
    for n in ('y_true', 'y_pred', 'y_evil'):
        name = os.path.join(args.output, f'{n}.npy')
        y = np.concatenate(locals()[n])
        locals()[n] = y
        np.save(name, y)

    # print stats
    acc_pred = (y_true == y_pred.argmax(1)).mean()
    acc_evil = (y_true == y_evil.argmax(1)).mean()
    print(f'Pred: {acc_pred:.3%}')
    print(f'Evil: {acc_evil:.3%}')
