import pickle
from argparse import ArgumentParser
from typing import List, Iterable

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from art.estimators.classification import PyTorchClassifier
from tqdm import tqdm
import numpy as np

from scaleadv.datasets import get_imagenet
from scaleadv.models import resnet50


def load(fname):
    return T.ToTensor()(Image.open(fname))


def load_l2(fname, budget: int):
    log = pickle.load(open(fname, 'rb'))
    for i, q, l2 in log:
        if q >= budget * 1000:
            return l2


def evaluate_one(ids: Iterable[int], budget: int):
    # all data
    pred = []
    l2_att = []
    l2_ada = []

    for id in tqdm(ids):
        # Load generic images
        src_large = load(f'static/bb_small/{id:02d}.{budget}k.src.png')
        src_small = NotImplemented
        adv_small = load(f'static/bb_small/{id:02d}.{budget}k.tar.png')
        y_src = dataset[id_list[id]][1]

        # Load attack images (hide)
        att_large_hide = load(f'static/bb_small/{id:02d}.{budget}k.att.png')
        att_small_hide = load(f'static/bb_small/{id:02d}.{budget}k.att_down.png')
        ada_large_hide = load(f'static/bb_small/{id:02d}.{budget}k.ada.png')
        ada_small_hide = load(f'static/bb_small/{id:02d}.{budget}k.ada_down.png')

        # Measure accuracy for hide
        x = torch.stack([adv_small, att_small_hide, ada_small_hide]).numpy()
        y_adv, y_att, y_ada = classifier.predict(x).argmax(1)
        pred.append((y_src, y_adv, y_att, y_ada))

        # Measure perturbation for hide
        l2_att_hide = (src_large - att_large_hide).norm()
        l2_ada_hide = (src_large - ada_large_hide).norm()

        # Measure perturbation for gen
        l2_att_gen = load_l2(f'static/bb_large/{id}.ratio_3.def_none.log', budget)
        l2_ada_gen = load_l2(f'static/bb_median/{id}.ratio_3.def_median.log', budget)

        l2_att.append((l2_att_hide, l2_att_gen))
        l2_ada.append((l2_ada_hide, l2_ada_gen))

    return list(map(np.array, [pred, l2_att, l2_ada]))


if __name__ == '__main__':
    p = ArgumentParser()
    args = p.parse_args()

    # Load data
    dataset = get_imagenet('val_3', None)
    id_list = pickle.load(open(f'static/meta/valid_ids.model_none.scale_3.pkl', 'rb'))[::4]

    # Load network
    class_network = resnet50(robust='none', normalize=True).eval().cuda()
    classifier = PyTorchClassifier(
        model=class_network,
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224),
        nb_classes=1000,
        clip_values=(0, 1),
    )

    p5, att5, ada5 = evaluate_one(range(0, 96, 2), 5)
    p10, att10, ada10 = evaluate_one(range(0, 96, 2), 10)
    import IPython as i; i.embed(using=False)

    # p == p[:, 0][..., None]
