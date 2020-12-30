import pickle
from argparse import ArgumentParser

import torch
import torchvision.transforms as T
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from scaleadv.datasets import get_imagenet
from scaleadv.models.resnet import IMAGENET_MODEL_PATH, resnet50

if __name__ == '__main__':
    p = ArgumentParser()
    _ = p.add_argument
    _('--model', default='none', type=str, choices=IMAGENET_MODEL_PATH.keys(), help='use robust model, optional')
    _('--scale', default=3, type=int, help='set a fixed scale ratio, unset to use the original size')
    args = p.parse_args()

    # Load data
    batch_size = 100
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    dataset = get_imagenet(f'val_{args.scale}', transform)
    loader = DataLoader(dataset, shuffle=False, num_workers=8, batch_size=batch_size)

    # Load network
    network = resnet50(args.model, normalize=True).eval().cuda()

    # Check
    id_list = []
    for i, (x, y) in tqdm(enumerate(loader)):
        p = network(x.cuda()).argmax(1).cpu()
        ids = i * batch_size + torch.nonzero(y == p).squeeze()
        id_list.extend(ids.tolist())

    path = f'static/meta/valid_ids.model_{args.model}.scale_{args.scale}.pkl'
    logger.info(f'Total: {len(dataset)}')
    logger.info(f'Correct: {len(id_list)}')
    logger.info(f'Saving id list to "{path}"')
    pickle.dump(id_list, open(path, 'wb'))
