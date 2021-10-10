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
import pandas as pd

from scaleadv.datasets import get_imagenet
from scaleadv.models import resnet50


def load(fname):
    return T.ToTensor()(Image.open(fname))


def load_l2(fname, budget: int):
    log = pickle.load(open(fname, 'rb'))
    for i, q, l2 in log:
        if q >= budget * 1000:
            return l2


def evaluate_one(ids: Iterable[int], budget: int, save: bool = True):
    fname = [
        f'static/hide-vs-gen/eval-pred-{budget}.npy',
        f'static/hide-vs-gen/eval-l2-{budget}.npy'
    ]
    if lazy:
        pred = np.load(fname[0])
        l2 = np.load(fname[1])
        return pred, l2

    pred, l2 = [], []
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
        l2_att_hide = (src_large - att_large_hide).norm().item() / 3
        l2_ada_hide = (src_large - ada_large_hide).norm().item() / 3

        # Measure perturbation for gen
        l2_att_gen = load_l2(f'static/bb_large/{id}.ratio_3.def_none.log', budget) / 3
        l2_ada_gen = load_l2(f'static/bb_median/{id}.ratio_3.def_median.log', budget) / 3

        l2.append((l2_att_hide, l2_att_gen, l2_ada_hide, l2_ada_gen))

    pred, l2 = map(np.array, [pred, l2])

    if save:
        np.save(fname[0], pred)
        np.save(fname[1], l2)
    return pred, l2


if __name__ == '__main__':
    l2_budgets = np.arange(0, 10, 1)
    dataset = get_imagenet('val_3', None)
    id_list = pickle.load(open(f'static/meta/valid_ids.model_none.scale_3.pkl', 'rb'))[::4]
    lazy = True

    # Load network
    if not lazy:
        class_network = resnet50(robust='none', normalize=True).eval().cuda()
        classifier = PyTorchClassifier(
            model=class_network,
            loss=nn.CrossEntropyLoss(),
            input_shape=(3, 224, 224),
            nb_classes=1000,
            clip_values=(0, 1),
        )

    # Collect all data
    data_all = []
    for b in [1, 3, 5, 7, 9, 10]:
        # load data for this budget
        # pred: samples of (y_src, y_adv, y_att, y_ada)
        # l2: samples of (l2_att_hide, l2_att_gen, l2_ada_hide, l2_ada_gen)
        pred, l2 = evaluate_one(range(0, 96, 2), b, save=True)

        # l2-vs-query (median over samples)
        l2_at_this_budget = np.median(l2, axis=0)

        # sar-vs-l2 (overall)
        l2_ok = l2[:, None, :] <= l2_budgets[..., None]  # dim = (samples, budgets, the four l2's ok)
        attack_ok = np.not_equal(pred[:, -3:], pred[:, 0, None])  # dim = (samples, the two hide's ok + the gen's ok)
        # sar-vs-l2 (non-adaptive, mean over samples)
        sar_att = np.mean(l2_ok[..., :2] & attack_ok[..., None, [1, 0]], axis=0) * 100  # dim = (budgets, the two attack)
        # sar-vs-l2 (adaptive, mean over samples)
        sar_ada = np.mean(l2_ok[..., 2:] & attack_ok[..., None, [2, 0]], axis=0) * 100  # dim = (budgets, the two attack)

        print(l2_at_this_budget)
        print(sar_att)
        print(sar_ada)
        input(f'Now {b}, next? ')
