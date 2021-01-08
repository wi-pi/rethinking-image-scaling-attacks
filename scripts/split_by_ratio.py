"""
This module implements an utility to split a dataset by scaling ratios w.r.t a given size.

Note:
  * We assume the dataset is ImageNet.

Our Results:
-- Ratio Stats
 2: 6555
 3: 599
 4: 185
 5: 160
 6: 61
 7: 40
 8: 42
 9: 27
10: 22
11: 7
12: 5
13: 2
16: 1
-- Split Stats
[2] 2-3: 6555
[3] 3-4: 599
[4] 4-5: 185
[5] 5-6: 160
[6] 6-10: 170
"""
import os
import pickle
import shutil
from collections import defaultdict
from operator import itemgetter
from pathlib import Path
from typing import Dict, List

import click
from loguru import logger
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from scaleadv.datasets import get_imagenet

INPUT_SIZE = 224
RATIO_INTERVALS = [  # [l, r)
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 10),
]
COUNTER_CACHE = Path('./static/meta/ratio_counter.pkl')
CounterType = Dict[int, List[int]]


def _count_ratio(dataset: ImageFolder, load: bool = False) -> CounterType:
    """Count the number of examples for each ratio (larger than 1)."""
    if load and COUNTER_CACHE.exists():
        logger.info(f'Loading counter from "{COUNTER_CACHE}"')
        return pickle.load(open(COUNTER_CACHE, 'rb'))

    # noinspection PyTypeChecker
    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=32)
    counter = defaultdict(list)
    with tqdm(loader, desc='Counting Ratios') as pbar:
        for i, (x, _) in enumerate(pbar):
            ratio = min(x.size) // INPUT_SIZE
            if ratio > 1:
                counter[ratio].append(i)
            stats = {f'[{k}]': len(v) for k, v in sorted(counter.items())}
            pbar.set_postfix(stats)

    os.makedirs(COUNTER_CACHE.parent, exist_ok=True)
    pickle.dump(counter, open(COUNTER_CACHE, 'wb'))
    return counter


def _split_ratio(counter: CounterType) -> CounterType:
    """Rearrange the counter by ratio intervals."""
    split = defaultdict(list)
    for ratio, imgs in counter.items():
        for L, R in RATIO_INTERVALS:
            if L <= ratio < R:
                split[L].extend(imgs)
                break
    return split


def _migrate(root: Path, save: Path, dataset: ImageFolder, split: CounterType):
    """Migrate dataset from root to save given a ratio split."""
    # Migrate data
    os.makedirs(save, exist_ok=True)

    # Move data
    for ratio, imgs in split.items():
        # Init ratio dir
        ratio_root = save / f'{root.name}_{ratio}'
        if ratio_root.exists():
            logger.warning(f'Ratio directory already exists: "{ratio_root.absolute()}"')
        os.makedirs(ratio_root, exist_ok=True)
        logger.info(f'Copying ratio {ratio} ({len(imgs)} files) to "{ratio_root}"')

        # Init class dirs
        for cname in dataset.classes:
            path = ratio_root / cname
            os.makedirs(str(path), exist_ok=True)

        # Copy data
        for src, _ in itemgetter(*imgs)(dataset.imgs):
            src = Path(src)
            dst = ratio_root / src.relative_to(root)
            shutil.copyfile(src, dst)


@click.command()
@click.option('--root', type=click.Path(exists=True), help='Path of the input dataset.')
@click.option('--save', type=click.Path(), help='Path of output ratio datasets.')
@click.option('--load', is_flag=True, help='Load cached counter.')
def main(root, save, load):
    """Split the dataset by ratios."""
    # Set root and save directory.
    root = Path(root)
    save = Path(save or root.parent.absolute())
    logger.info(f'Input dataset: "{root.absolute()}"')
    logger.info(f'Output dataset: "{save.absolute()}"')

    # Count ratios
    dataset = get_imagenet('val')
    counter = _count_ratio(dataset, load=load)
    print('-- Ratio Stats')
    for k, v in sorted(counter.items()):
        print(f'{k:>2d}: {len(v)}')

    # Split
    split = _split_ratio(counter)
    print('-- Split Stats')
    for (l, r), (ratio, imgs) in zip(RATIO_INTERVALS, sorted(split.items())):
        print(f'[{ratio}] {l}-{r}: {len(imgs)}')

    # Migrate data
    _migrate(root, save, dataset, split)


if __name__ == '__main__':
    main()
