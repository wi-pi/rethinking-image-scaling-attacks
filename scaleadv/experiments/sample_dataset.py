import os
import pickle
from pathlib import Path

from torch.utils.data import DataLoader

from scaleadv.datasets.imagenet import create_dataset

STORE = Path('static/datasets/imagenet-600')
DUMP = 'data-600.pkl'
INPUT_SIZE = 224
TOTAL_IMAGES = 600
RATIO_IMAGES = 120
RATIO_INTERVALS = [  # [l, r)
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 7),
    (7, 10)
]

if __name__ == '__main__':
    # load data
    dataset = create_dataset(transform=None)
    loader = DataLoader(dataset, batch_size=None, shuffle=True, num_workers=8)

    # get data indices
    if os.path.exists(DUMP):
        DATA = pickle.load(open(DUMP, 'rb'))
    else:
        DATA = {interval: [] for interval in RATIO_INTERVALS}
        for index, x, y in loader:
            # collect if possible
            ratio = min(x.size) / INPUT_SIZE
            for (L, R), images in DATA.items():
                if L <= ratio < R and len(images) < RATIO_IMAGES:
                    images.append(index)
            # check stop
            total = list(map(len, DATA.values()))
            if sum(total) >= TOTAL_IMAGES:
                break
            print(' '.join([f'{v}/{RATIO_IMAGES}' for v in total]), end='\r')
        print()
        pickle.dump(DATA, open(DUMP, 'wb'))

    # prepare dirs
    os.makedirs(str(STORE), exist_ok=True)
    for (L, _) in RATIO_INTERVALS:
        ratio_path = STORE / str(L).replace('.', '_')
        for i in range(1000):
            path = ratio_path / f'{i:03d}'
            os.makedirs(str(path), exist_ok=True)

    # store data
    for (L, _), indices in DATA.items():
        # size = int(INPUT_SIZE * (L + 0.5))
        for index in indices:
            src, y = dataset.imgs[index]
            dst = STORE / str(L).replace('.', '_') / f'{y:03d}'
            os.system(f'cp "{src}" "{str(dst)}"')
            # _, x, y = dataset[index]
            # dst = STORE / str(L).replace('.', '_') / f'{y:03d}' / Path(dataset.imgs[index][0]).name
            # x.resize((size, size)).save(str(dst))
