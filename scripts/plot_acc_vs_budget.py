import pickle
from argparse import ArgumentParser
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torchvision.transforms as T

from scaleadv.datasets import get_imagenet
from scaleadv.datasets.transforms import Align
from scaleadv.evaluate.utils import DataManager
from scaleadv.models.resnet import IMAGENET_MODEL_PATH, resnet50
from scaleadv.scaling import ScalingLib, ScalingAlg, ScalingAPI
from scaleadv.utils import set_ccs_font, get_id_list_by_ratio

if __name__ == '__main__':
    p = ArgumentParser()
    _ = p.add_argument
    # Input args
    _('eval', type=str, choices=('hide', 'generate'), help='which attack to evaluate')
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

    # Load data
    transform = T.Compose([Align(224, args.scale), T.ToTensor(), lambda x: np.array(x)[None, ...]])
    dataset = get_imagenet(f'val_{args.scale}', transform)

    # Load networks
    src_shape = (224 * args.scale, 224 * args.scale)
    inp_shape = (224, 224)
    scaling_api = ScalingAPI(src_shape, inp_shape, args.lib, args.alg)
    class_network = resnet50(args.model, normalize=True).eval().cuda()

    # Load utils
    id_list = pickle.load(open(f'static/meta/valid_ids.model_{args.model}.scale_{args.scale}.pkl', 'rb'))
    id_list = get_id_list_by_ratio(id_list, args.scale)
    eps_list = list(range(args.left, args.right, args.step))
    dm = DataManager(scaling_api)
    get_adv_data = lambda e: [dm.load_adv(i, e) for i in id_list]
    get_att_data = lambda e, d: [dm.load_att(i, e, d, args.eval) for i in id_list]

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
                    pert.append(att_stat['att']['L2'] / args.scale)

            acc, pert = map(np.mean, [acc, pert])
            acc_list[f'att_{defense}'].append(acc * 100)
            pert_list[f'att_{defense}'].append(pert)

        if args.eval == 'generate':
            pert_list[f'att_{defense}'] = eps_list

    # plot
    set_ccs_font(14)
    plt.figure(tight_layout=True)
    plt.plot(eps_list, acc_list['adv'], marker='o', ms=2, label='PGD Attack')
    plt.plot(pert_list['att_none'], acc_list['att_none'], marker='o', ms=2, label='Scale-Adv Attack (none)')
    plt.plot(pert_list['att_median'], acc_list['att_median'], marker='o', ms=2, label='Scale-Adv Attack (median)')
    plt.plot(pert_list['att_uniform'], acc_list['att_uniform'], marker='o', ms=2, label='Scale-Adv Attack (random)')
    plt.xlim(-0.5, args.right + 1)
    plt.xticks(list(range(0, args.right + 1, 2)))
    plt.ylim(-2, 102)
    plt.yticks(list(range(0, 101, 10)))
    plt.legend()
    plt.savefig(f'acc-{args.eval}.{args.scale}.pdf')
