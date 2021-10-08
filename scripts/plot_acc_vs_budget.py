import pickle
from argparse import ArgumentParser
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torchvision.transforms as T

from scaleadv.datasets import get_imagenet
from scaleadv.datasets.transforms import Align
from scaleadv.evaluate.utils import DataManager
from scaleadv.models.resnet import IMAGENET_MODEL_PATH
from scaleadv.scaling import ScalingLib, ScalingAlg, ScalingAPI, str_to_alg, str_to_lib
from scaleadv.utils import set_ccs_font, get_id_list_by_ratio

BLUE, ORANGE, GREEN, RED = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]


def get_acc_pert(lib, alg, scale, eva):
    # Load data
    transform = T.Compose([Align(224, scale), T.ToTensor(), lambda x: np.array(x)[None, ...]])
    dataset = get_imagenet(f'val_{scale}', transform)

    # Load networks
    src_shape = (224 * scale, 224 * scale)
    inp_shape = (224, 224)
    scaling_api = ScalingAPI(src_shape, inp_shape, lib, alg)

    # Load utils
    id_list = pickle.load(open(f'static/meta/valid_ids.model_{args.model}.scale_{scale}.pkl', 'rb'))
    id_list = get_id_list_by_ratio(id_list, scale)
    eps_list = list(range(args.left, args.right, args.step))
    dm = DataManager(scaling_api)
    get_adv_data = lambda e: [dm.load_adv(i, e) for i in id_list]
    get_att_data = lambda e, d: [dm.load_att(i, e, d, eva) for i in id_list]

    acc_list = defaultdict(list)
    pert_list = defaultdict(list)
    # get acc-vs-eps for adversarial attack
    for eps in eps_list:
        data = get_adv_data(eps)
        acc = []
        for stat, i in zip(data, id_list):
            if stat['src']['Y'] != dataset.targets[i]:
                continue
            acc.append(stat['adv']['Y'] == dataset.targets[i])
        acc_list['adv'].append(np.mean(acc) * 100)

    # get acc-vs-eps for scaling attack
    for defense, field in zip(['none', 'median', 'uniform'], ['Y', 'Y_MED', 'Y_RND']):
        for eps in eps_list:
            adv_data = get_adv_data(eps)
            att_data = get_att_data(eps, defense)
            acc, pert = [], []
            for adv_stat, att_stat, i in zip(adv_data, att_data, id_list):
                if adv_stat['src']['Y'] != dataset.targets[i]:
                    continue
                # NOTE: View hide as a set of correctly-predicted examples:
                #   1) small adv that is unsuccessful;
                #   2) big att for successful adv.
                if att_stat is None:
                    acc.append(adv_stat['adv']['Y'] == dataset.targets[i])
                    pert.append(adv_stat['adv']['L2'])
                else:
                    acc.append(scipy.stats.mode(att_stat['att'][field])[0].item() == dataset.targets[i])
                    pert.append(att_stat['att']['L2'] / scale)

            acc, pert = map(np.mean, [acc, pert])
            acc_list[f'att_{defense}'].append(acc * 100)
            pert_list[f'att_{defense}'].append(pert)

        if eva == 'generate':
            pert_list[f'att_{defense}'] = eps_list

    return acc_list, pert_list, eps_list


def plot_all():
    acc_list, pert_list, eps_list = get_acc_pert(args.lib, args.alg, args.scale, args.eval)
    set_ccs_font(10)
    plt.figure(figsize=(3, 3), constrained_layout=True)
    plt.plot(pert_list['att_none'], acc_list['att_none'], marker='o', ms=4, lw=1.5, c=GREEN, label='PGD Attack (scaling)')
    plt.plot(pert_list['att_median'], acc_list['att_median'], marker='^', ms=4, lw=1.5, c=ORANGE, label='PGD Attack (median)')
    plt.plot(pert_list['att_uniform'], acc_list['att_uniform'], marker='s', ms=4, lw=1.5, c=RED, label='PGD Attack (random)')
    plt.plot(eps_list, acc_list['adv'], marker='D', ms=4, lw=1.5, c='k', label='PGD Attack (vanilla)')
    plt.plot(pert_list['att_uniform'], acc_list['att_uniform'], marker='s', ms=4, lw=1.5, c=RED)
    plt.xlim(-0.5, args.right + 0.5)
    plt.xticks(list(range(0, args.right + 1, 2)), fontsize=12)
    plt.xlabel(r'Perturbation Budget ($\ell_2$)')
    plt.ylim(-2, 102)
    plt.yticks(list(range(0, 101, 20)), fontsize=12)
    plt.ylabel('Accuracy (%)')
    plt.legend(borderaxespad=0.5)
    plt.grid(True)
    plt.savefig(f'acc-{args.eval}.{args.scale}.pdf')


def plot_hide_generate():
    # init
    set_ccs_font(10)
    plt.figure(figsize=(3, 3), constrained_layout=True)

    # hide
    # acc_list, pert_list, eps_list = get_acc_pert(args.lib, args.alg, args.scale, 'hide')
    # plt.plot(pert_list['att_none'], acc_list['att_none'], marker='o', ms=4, lw=1.5, c=RED, label='Scale-Adv (Hide)')

    # generate
    acc_list, pert_list, eps_list = get_acc_pert(args.lib, args.alg, args.scale, 'generate')
    plt.plot(pert_list['att_none'], acc_list['att_none'], marker='o', ms=4, lw=1.5, ls='--', c=GREEN, label='PGD Attack (scaling)')
    plt.plot(eps_list, acc_list['adv'], marker='D', ms=4, lw=1.5, c='k', label='PGD Attack (vanilla)')
    plt.plot(pert_list['att_none'], acc_list['att_none'], marker='o', ms=4, lw=1.5, ls='--', c=GREEN)

    # final
    plt.xlim(-0.5, args.right + 0.5)
    plt.xticks(list(range(0, args.right + 1, 2)), fontsize=12)
    plt.xlabel(r'Perturbation Budget (scaled $\ell_2$)')
    plt.ylim(-2, 102)
    plt.yticks(list(range(0, 101, 20)), fontsize=12)
    plt.ylabel('Accuracy (%)')
    plt.legend(borderaxespad=0.5)
    plt.grid(True)
    plt.savefig(f'acc-all.{args.lib}.{args.alg}.{args.scale}.pdf')


if __name__ == '__main__':
    p = ArgumentParser()
    _ = p.add_argument
    # Input args
    _('eval', type=str, choices=('hide', 'generate', 'all'), help='which attack to evaluate')
    _('--model', default='none', type=str, choices=IMAGENET_MODEL_PATH.keys(), help='use robust model, optional')
    # Scaling args
    _('--lib', default='cv', type=str, choices=str_to_lib.keys(), help='scaling libraries')
    _('--alg', default='linear', type=str, choices=str_to_alg.keys(), help='scaling algorithms')
    _('--scale', default=3, type=int, help='set a fixed scale ratio, unset to use the original size')
    # Adversarial attack args
    _('-l', '--left', default=1, type=int)
    _('-r', '--right', default=21, type=int)
    _('-s', '--step', default=1, type=int)
    args = p.parse_args()

    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False

    if args.eval == 'all':
        plot_hide_generate()
    else:
        plot_all()
