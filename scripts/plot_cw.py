import pickle
from argparse import ArgumentParser
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T

from scaleadv.datasets import get_imagenet
from scaleadv.datasets.transforms import Align
from scaleadv.evaluate.utils import DataManager
from scaleadv.models.resnet import IMAGENET_MODEL_PATH
from scaleadv.scaling import ScalingLib, ScalingAlg, ScalingAPI
from scaleadv.utils import set_ccs_font, get_id_list_by_ratio

BLUE, ORANGE, GREEN, RED = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]


def calc_sar(acc_list, pert_list, budget_list):
    """Given a budget (max pert), how many data is adversarial and within the perturbation budget?
    """
    acc, pert, budget = map(np.array, [acc_list, pert_list, budget_list])
    ok = (1 - acc) * (pert <= budget[..., None])
    return ok.mean(axis=1) * 100


def get_acc_pert(lib, alg, scale, budget=None):
    # Load data
    transform = T.Compose([Align(224, scale), T.ToTensor(), lambda x: np.array(x)[None, ...]])
    dataset = get_imagenet(f'val_{scale}', transform)

    # Load networks
    src_shape = (224 * scale, 224 * scale)
    inp_shape = (224, 224)
    scaling_api = ScalingAPI(src_shape, inp_shape, lib, alg)

    # Load utils
    id_list = pickle.load(open(f'static/meta/valid_ids.model_{args.model}.scale_{scale}.pkl', 'rb'))
    id_list = get_id_list_by_ratio(id_list, scale)[::2]
    id_list = [id_list[i] for i in range(120)]
    eps_list = list(range(args.left, args.right + 1, args.step))
    dm = DataManager(scaling_api)
    get_adv_data = lambda e: [dm.load_adv(i, e) for i in id_list]
    get_att_data = lambda e, d: [dm.load_att(i, e, d, 'generate') for i in id_list]

    acc_list = defaultdict(list)
    pert_list = defaultdict(list)
    sar_list = defaultdict(list)

    # get acc/pert-vs-eps for adversarial attack
    for eps in eps_list:
        data = get_adv_data(eps)
        acc, pert = [], []
        for stat, i in zip(data, id_list):
            if stat['src']['Y'][0] != dataset.targets[i]:
                continue
            acc.append(stat['adv']['Y'][0] == dataset.targets[i])
            pert.append(stat['adv']['L2'])
        acc_list['adv'].append(np.mean(acc) * 100)
        pert_list['adv'].append(np.median(pert))
        if budget is not None:
            sar_list['adv'].append(calc_sar(acc, pert, budget))

    # get acc/pert-vs-eps for scaling attack
    for defense, field in zip(['none', 'median'], ['Y', 'Y_MED']):
        for eps in eps_list:
            adv_data = get_adv_data(eps)
            att_data = get_att_data(eps, defense)
            acc, pert = [], []
            for adv_stat, att_stat, i in zip(adv_data, att_data, id_list):
                if adv_stat['src']['Y'][0] != dataset.targets[i]:
                    continue
                if att_stat['att'][field][0] == dataset.targets[i]:
                    acc.append(True)
                    pert.append(np.inf)
                else:
                    acc.append(False)
                    pert.append(att_stat['att']['L2'] / scale)

            acc_list[f'att_{defense}'].append(np.mean(acc) * 100)
            pert_list[f'att_{defense}'].append(np.median(pert))
            if budget is not None:
                sar_list[f'att_{defense}'].append(calc_sar(acc, pert, budget))

    ret = acc_list, pert_list, eps_list, sar_list
    return ret[:-1] if budget is None else ret


def plot_all():
    acc_list, pert_list, eps_list = get_acc_pert(args.lib, args.alg, args.scale)
    set_ccs_font(10)

    plt.figure(figsize=(3, 3), constrained_layout=True)
    plt.plot(eps_list, pert_list['att_none'], marker='o', ms=4, lw=1.5, c=GREEN, label='Scale-Adv (none)')
    plt.plot(eps_list, pert_list['att_median'], marker='^', ms=4, lw=1.5, c=ORANGE, label='Scale-Adv (median)')
    plt.plot(eps_list, pert_list['adv'], marker='D', ms=4, lw=1.5, c='k', label='C&W Attack')
    plt.xlim(-0.5, args.right + 0.5)
    plt.xticks(list(range(0, args.right + 1, 2)), fontsize=12)
    plt.xlabel(r'Confidence ($\kappa$)')
    plt.ylabel(r'Distance ($\ell_2$)')
    plt.yscale('log')
    plt.legend(borderaxespad=0.5)
    plt.grid(True)
    plt.savefig(f'cw-pert-vs-kappa.pdf')


def plot_sar_vs_pert():
    budget = np.arange(0, 21)
    acc_list, pert_list, eps_list, sar_list = get_acc_pert(args.lib, args.alg, args.scale, budget)
    set_ccs_font(10)

    plt.figure(figsize=(3, 3), constrained_layout=True)
    text_kwargs = dict(fontsize=8, rotation_mode='anchor', bbox=dict(fc='white', ec='none', pad=0),
                       transform_rotates_text=True)

    def _pp(tag, kappa, c, ls, label, pos):
        plt.plot(budget, sar_list[tag][kappa], ls=ls, lw=1.5, c=c, label=label)
        dy = sar_list[tag][kappa][pos + 1] - sar_list[tag][kappa][pos]
        rot = np.degrees(np.arctan2(dy, 1))
        plt.text(budget[pos], sar_list[tag][kappa][pos], rf'$\kappa={kappa}$', c=c, rotation=rot, **text_kwargs)

    _pp('att_none', 0, 'k', '-', 'Scale-Adv (none)', 3)
    _pp('att_median', 0, 'k', '--', 'Scale-Adv (median)', 3)
    _pp('adv', 0, 'k', ':', 'C&W Attack', 3)
    _pp('att_none', 5, 'b', '-', None, 11)
    _pp('att_median', 5, 'b', '--', None, 11)
    _pp('adv', 5, 'b', ':', None, 11)
    _pp('att_none', 10, 'r', '-', None, 17)
    _pp('att_median', 10, 'r', '--', None, 17)
    _pp('adv', 10, 'r', ':', None, 17)
    plt.xlim(-0.5, 20.5)
    plt.xticks(list(range(0, 21, 5)), fontsize=12)
    plt.ylim(-2, 102)
    plt.yticks(list(range(0, 101, 10)), fontsize=12)
    plt.xlabel(r'Perturbation Budget ($\ell_2$)')
    plt.ylabel('Success Rate (%)')
    plt.legend(borderaxespad=0.5, loc='lower right', fontsize=10)
    plt.grid(True)
    plt.savefig(f'cw-sar-vs-pert.pdf')


if __name__ == '__main__':
    p = ArgumentParser()
    _ = p.add_argument
    # Input args
    _('--model', default='none', type=str, choices=IMAGENET_MODEL_PATH.keys(), help='use robust model, optional')
    # Scaling args
    _('--lib', default='cv', type=str, choices=ScalingLib.names(), help='scaling libraries')
    _('--alg', default='linear', type=str, choices=ScalingAlg.names(), help='scaling algorithms')
    _('--scale', default=3, type=int, help='set a fixed scale ratio, unset to use the original size')
    # Adversarial attack args
    _('-l', '--left', default=1, type=int)
    _('-r', '--right', default=21, type=int)
    _('-s', '--step', default=1, type=int)
    args = p.parse_args()

    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    plot_all()
    plot_sar_vs_pert()
