import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.utils import DatasetHelper
from src.models import imagenet_resnet50, celeba_resnet34


def main(args):
    # Load dataset
    dataset = DatasetHelper(name=args.dataset, scale=1, base=224, valid_samples_only=False, min_scale=3)
    loader = DataLoader(dataset, batch_size=args.batch, num_workers=32)

    # Load network
    match args.dataset:
        case 'imagenet':
            model = imagenet_resnet50('nature', normalize=True)
            filename = args.output / 'valid_ids.imagenet.nature.npy'
        case 'celeba':
            model = celeba_resnet34(num_classes=11, binary_label=6, ckpt='nature')
            filename = args.output / 'valid_ids.celeba.Mouth_Slightly_Open.npy'
        case _:
            raise NotImplementedError(args.dataset)

    # Prepare
    model = model.eval().cuda()

    # Eval loop
    valid_ids = []
    dataset_ids = torch.tensor(dataset.id_list)
    for i, (x, y) in enumerate(tqdm(loader, desc='Evaluate')):
        with torch.no_grad():
            y_pred = model(x.cuda()).argmax(1).cpu()

        valid_ids_of_this_batch = torch.eq(y_pred, y).nonzero().squeeze() + i * args.batch
        valid_ids += dataset_ids[valid_ids_of_this_batch].tolist()

    # Summary
    logger.info(f'Correct: {len(valid_ids)}')
    logger.info(f'Accuracy: {len(valid_ids)} / {len(dataset)} ({len(valid_ids) / len(dataset):.2%})')

    # Dump
    logger.info(f'Saving id list to "{filename}"')
    os.makedirs(args.output, exist_ok=True)
    np.save(filename, valid_ids)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=['imagenet', 'celeba'], required=True)
    parser.add_argument('-b', '--batch', type=int, default=256)
    parser.add_argument('-o', '--output', type=Path, default='static/meta/')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stderr, level='INFO')
    main(parse_args())
