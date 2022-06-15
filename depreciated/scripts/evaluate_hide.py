import pickle
import matplotlib.pyplot as plt
from typing import Iterable

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from art.estimators.classification import PyTorchClassifier
from tqdm import tqdm
import numpy as np

import matplotlib as mpl
from depreciated.scaleadv import get_imagenet
from depreciated.scaleadv import resnet50
from depreciated.scaleadv.utils import set_ccs_font


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

        # Load perturbation for hsj
        l2_hsj = load_l2(f'static/bb_large/{id}.ratio_1.def_none.log', budget)

        l2.append((l2_att_hide, l2_att_gen, l2_ada_hide, l2_ada_gen, l2_hsj))

    pred, l2 = map(np.array, [pred, l2])

    if save:
        np.save(fname[0], pred)
        np.save(fname[1], l2)
    return pred, l2


def plot_pert_vs_queries(l2_hide, l2_gen, l2_hsj, tag):
    set_ccs_font(10)
    plt.figure(figsize=(3, 3), constrained_layout=True)
    plt.plot(budget_list, l2_gen, marker='o', ms=4, lw=1.5, c=GREEN, label=f'Joint Attack ({tag})')
    plt.plot(budget_list, l2_hide, marker='^', ms=4, lw=1.5, c=ORANGE, label=f'Sequential Attack ({tag})')
    # plt.plot(budget_list, l2_hsj, ls='--', ms=4, lw=1.5, c='k', label='HSJ Attack')
    plt.legend(borderaxespad=0.5)
    plt.yscale('log')
    plt.xticks(list(range(0, 26, 5)), [f'{i}K' for i in range(0, 26, 5)])
    plt.xlabel('Number of Queries (#)')
    plt.ylabel(r'Perturbation (scaled $\ell_2$)')
    plt.grid(True)
    plt.savefig(f'gen-vs-hide-l2-vs-queries-{tag}.pdf')


def plot_sar_vs_pert(sar_hide, sar_gen, tag, ips, x_step):
    set_ccs_font(10)
    plt.figure(figsize=(3, 3), constrained_layout=True)
    text_kwargs = dict(fontsize=8, rotation_mode='anchor', bbox=dict(fc='white', ec='none', pad=0),
                       transform_rotates_text=True)

    def _pp(sar, query, c, ls, label, pos):
        plt.plot(l2_budgets, sar, ls=ls, lw=1.5, c=c, label=label)
        rot = np.degrees(np.arctan2(sar[pos + 1] - sar[pos], 0.1))
        plt.text(l2_budgets[pos], sar[pos], f'{query}K', c=c, rotation=rot, **text_kwargs)

    # plot query 1K
    for ind, (i, p1, p2) in enumerate(ips):
        _pp(sar_gen[i], budget_list[i], 'kbrg'[ind], '-', None if ind else 'Joint Attack', p1)
        _pp(sar_hide[i], budget_list[i], 'kbrg'[ind], '--', None if ind else 'Sequential Attack', p2)

    plt.xlim(-0.05, l2_max + .05)
    plt.xticks(np.arange(0, l2_max + .1, x_step), fontsize=12)
    plt.ylim(-2, 102)
    plt.yticks(list(range(0, 101, 20)), fontsize=12)
    plt.xlabel(r'Perturbation Budget (scaled $\ell_2$)')
    plt.ylabel('Success Attack Rate (%)')
    loc = 'upper right' if tag == 'scaling' else 'upper right'
    plt.legend(borderaxespad=0.5, loc=loc, fontsize=10)
    plt.grid(True)
    plt.savefig(f'gen-vs-hide-sar-vs-pert-{tag}.pdf')


if __name__ == '__main__':
    l2_max = 50
    l2_budgets = np.arange(0, l2_max + 1, 1)
    dataset = get_imagenet('val_3', None)
    id_list = pickle.load(open(f'static/meta/valid_ids.model_none.scale_3.pkl', 'rb'))[::4]
    budget_list = [1, 3, 5, 7, 9, 10, 15, 20]
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
    l2_all, sar_att_all, sar_ada_all = [], [], []
    for b in budget_list:
        # load data for this budget
        # pred: samples of (y_src, y_adv, y_att, y_ada)
        # l2: samples of (l2_att_hide, l2_att_gen, l2_ada_hide, l2_ada_gen)
        pred, l2 = evaluate_one(range(0, 96, 2), b, save=True)

        # l2-vs-query (median over samples)
        l2_at_this_budget = np.median(l2, axis=0)
        l2_all.append(l2_at_this_budget)

        # sar-vs-l2 (overall)
        l2_ok = l2[:, None, :] <= l2_budgets[..., None]  # dim = (samples, budgets, the four l2's ok)
        attack_ok = np.not_equal(pred[:, -3:], pred[:, 0, None])  # dim = (samples, the two hide's ok + the gen's ok)
        # sar-vs-l2 (non-adaptive, mean over samples)
        sar_att = np.mean(l2_ok[..., 0:2] & attack_ok[..., None, [1, 0]], axis=0) * 100  # dim = (budgets, the two attack)
        sar_att_all.append(sar_att)
        # sar-vs-l2 (adaptive, mean over samples)
        sar_ada = np.mean(l2_ok[..., 2:4] & attack_ok[..., None, [2, 0]], axis=0) * 100  # dim = (budgets, the two attack)
        sar_ada_all.append(sar_ada)

    l2_all = np.array(l2_all)
    sar_att_all = np.array(sar_att_all)  # dim = (budgets, l2_budgets, hide + gen)
    sar_ada_all = np.array(sar_ada_all)  # dim = (budgets, l2_budgets, hide + gen)

    # Plot!
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    BLUE, ORANGE, GREEN, RED = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]

    # the two plots below are not that informative
    # plot_pert_vs_queries(l2_hide=l2_all[:, 0], l2_gen=l2_all[:, 1], l2_hsj=l2_all[:, -1], tag='scaling')
    # plot_pert_vs_queries(l2_hide=l2_all[:, 2], l2_gen=l2_all[:, 3], l2_hsj=l2_all[:, -1], tag='median')

    # this plot is more informative when: l2_max=5, l2_budgets with step 0.1
    if l2_max == 5:
        ips = zip([0, 2, 7], [40, 20, 10], [40, 20, 10])
        plot_sar_vs_pert(sar_hide=sar_att_all[..., 0], sar_gen=sar_att_all[..., 1], tag='scaling', ips=ips, x_step=1)

    # this plot is more informative when: l2_max=50, l2_budgets with step 1.0
    if l2_max == 50:
        ips = zip([0, 1, 2, 7], [15, 6, 10, 10], [28, 30, 35, 40])
        plot_sar_vs_pert(sar_hide=sar_ada_all[..., 0], sar_gen=sar_ada_all[..., 1], tag='median', ips=ips, x_step=10)
