import concurrent.futures
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries
from sklearn.manifold import TSNE
from tqdm import tqdm

from scaleadv.datasets.imagenet import create_dataset
from scaleadv.models.layers import RandomPool2d

OUTPUT = f'static/results/experiments/random_dist'
os.makedirs(OUTPUT, exist_ok=True)


def get_embeddings(x: torch.Tensor, pooling: RandomPool2d, n: int = 200, d: int = 2):
    y = pooling(x.repeat(n, 1, 1, 1)).reshape(n, -1).numpy()
    z = TSNE(n_components=d).fit_transform(y).T
    return z


def task(i):
    ind, x, tag = dataset[i]
    z = get_embeddings(x * 255, pooling, n=200, d=2)
    return ind, z, tag


if __name__ == '__main__':
    # args
    p = ArgumentParser()
    p.add_argument('-k', '--kernel', type=int, required=True, metavar='K', help='kernel width')
    p.add_argument('-s', '--seed', type=int, default=100, metavar='S', help='random seed')
    args = p.parse_args()

    # load data
    dataset = create_dataset(transform=T.ToTensor())
    pooling = RandomPool2d(args.kernel, 1, args.kernel // 2)

    # get embeddings
    N = 5
    np.random.seed(args.seed)
    ids = np.random.randint(0, len(dataset), N)
    with concurrent.futures.ProcessPoolExecutor() as exe:
        output = list(tqdm(exe.map(task, ids), total=N))

    # plot all
    plt.figure()
    for ind, z, tag in output:
        plt.scatter(*z, s=1, label=f'{ind} ({tag})')
    plt.legend()
    plt.savefig(f'{OUTPUT}/all-{args.kernel}.pdf')

    # plt separate
    for ind, z, _ in output:
        plt.figure()
        plt.scatter(*z, s=1)
        plt.savefig(f'{OUTPUT}/{ind}-{args.kernel}.pdf')
