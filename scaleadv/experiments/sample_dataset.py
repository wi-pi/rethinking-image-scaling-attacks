"""
This module splits ImageNet into several scale ratios.

Ratio Stats:
2 6555
3 599
4 185
5 160
6 61
7 40
8 42
9 27
10 22
11 7
12 5
13 2
16 1

Split Stats:
2-3: 6555
3-4: 599
4-5: 185
5-6: 160
6-10: 170
"""
import os
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm, trange

from scaleadv.datasets.imagenet import create_dataset

STORE = Path('static/datasets/imagenet-ratio')
DUMP = 'data-count.pkl'
INPUT_SIZE = 224

RATIO_INTERVALS = [  # [l, r)
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 10),
]


def count_ratio(data: ImageFolder, load: bool = False):
    loader = DataLoader(data, batch_size=None, shuffle=False, num_workers=32)
    if load and os.path.exists(DUMP):
        return pickle.load(open(DUMP, 'rb'))

    counter = defaultdict(list)
    with tqdm(loader, desc='Loading Dataset') as pbar:
        for i, (x, _) in enumerate(pbar):
            ratio = min(x.size) // INPUT_SIZE
            if ratio > 1:
                counter[ratio].append(i)
            if i % 1000 == 0:
                pickle.dump(counter, open(DUMP, 'wb'))
            pbar.set_postfix({f'[{k}]': len(v) for k, v in sorted(counter.items())})
    pickle.dump(counter, open(DUMP, 'wb'))
    return counter


def split_ratio(counter: Dict):
    split = defaultdict(list)
    for ratio, imgs in counter.items():
        for l, r in RATIO_INTERVALS:
            if l <= ratio < r:
                split[f'{l}'].extend(imgs)
                break
    print('Split Results')
    for k, v in split.items():
        print(k, len(v))
    return split


def copy_data(root: Path, data: ImageFolder, split: Dict):
    for k, v in split.items():
        # init dir
        for i in trange(1000, desc=f'Initialize {k}'):
            path = root / k / f'{i:03d}'
            os.makedirs(str(path), exist_ok=True)
        # copy data
        for i in tqdm(v, desc=f'Copy {k}'):
            src, y = data.imgs[i]
            src = Path(src)
            dst = root / k / f'{y:03d}' / src.name
            shutil.copyfile(src, dst)


if __name__ == '__main__':
    dataset = create_dataset(transform=None)
    counter = count_ratio(dataset, load=True)
    split = split_ratio(counter)
    copy_data(STORE, dataset, split)
