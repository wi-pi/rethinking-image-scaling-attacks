"""
This module implements experiments that compare Accuracy vs Attack Budget.
"""
import os
import pickle
from collections import defaultdict
from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from scaleadv.tests.scale_adv import *


def get_dataset_by_ratio(ratio: int, nb_data: Optional[int] = None):
    root = f'static/datasets/imagenet-ratio/{ratio}'
    t = T.Compose([
        lambda img: resize_to_224x(img, scale=ratio, square=True),
        T.ToTensor(),
        lambda x: x.numpy(),
    ])
    dataset = create_dataset(root=root, transform=t)
    if nb_data is not None:
        tot = len(dataset)
        dataset = Subset(dataset, list(range(0, tot, tot // nb_data))[:nb_data])
    return dataset


class AccBudget(object):
    # PGD params
    NORM = 2
    STEP = 30
    BIG_STEP = 100
    NB_DATA = 100

    def __init__(self, class_net: nn.Module, target: int, lib: str, algo: str, tag: str):
        self.class_net = class_net
        self.target = target
        self.lib = LIB_TYPE[lib]
        self.algo = ALGO_TYPE[algo]
        self.tag = tag
        self.saved_results = defaultdict(list)

    def eval_ratio_eps(self, ratio: int, eps: int, defense: str = 'none', mode: str = None,
                       dump: bool = True, load: bool = False):
        fname = f'eval-dataset.{ratio}.{eps:03d}.{defense}.{mode}.pkl'
        if load:
            if os.path.exists(fname):
                print(fname)
                return pickle.load(open(fname, 'rb'))
            return None

        # Load data
        dataset = get_dataset_by_ratio(ratio, nb_data=self.NB_DATA)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
        src_size = (224 * ratio,) * 2 + (3,)
        inp_size = INPUT_SHAPE_PIL

        # Load scaling algorithm
        scaling = ScalingGenerator.create_scaling_approach(src_size, inp_size, self.lib, self.algo)
        mask = get_mask_from_cl_cr(scaling.cl_matrix, scaling.cr_matrix)

        # Get pooling layer
        kernel = src_size[0] // inp_size[0] * 2 - 1
        pooling_args = kernel, 1, kernel // 2, mask
        pooling = POOLING[defense](*pooling_args)
        nb_samples = 1 if mode is None else 20

        # Get networks
        scale_net = ScaleNet(scaling.cl_matrix, scaling.cr_matrix).eval()
        class_net = self.class_net.eval()
        if nb_samples > 99:
            class_net = BalancedDataParallel(FIRST_GPU_BATCH, class_net)
        scale_net = scale_net.cuda()
        class_net = class_net.cuda()

        # Get art's proxy
        classify = PyTorchClassifier(class_net, nn.CrossEntropyLoss(), INPUT_SHAPE_NP, NUM_CLASSES, clip_values=(0, 1))

        # Get art's attack
        eps_step = 2.5 * eps / self.STEP
        targeted = self.target is not None
        adv_attack = IndirectPGD(classify, self.NORM, eps, eps_step, self.STEP, targeted=targeted, verbose=False)

        # Get scaling attack
        scl_attack = ScaleAttack(scale_net, class_net, pooling)
        big_eps = eps * ratio
        big_eps_step = big_eps * 30. / self.BIG_STEP
        attack_args = dict(norm=self.NORM, eps=big_eps, eps_step=big_eps_step, max_iter=self.BIG_STEP,
                           targeted=targeted, batch_size=NUM_SAMPLES_PROXY, verbose=False)

        # Eval all data
        e = Evaluator(scale_net, class_net, pooling_args, nb_samples=nb_samples)
        desc = f'Evaluate (ratio {ratio}, eps {eps}, defense {defense})'
        data = []
        with tqdm(loader, desc=desc) as pbar:
            for i, (src, y) in enumerate(pbar):
                # get input
                inp = scale_net(src.cuda()).cpu().numpy()
                src = src.cpu().numpy()
                # get adv
                y_tgt = np.eye(NUM_CLASSES, dtype=np.int)[None, self.target if targeted else y]
                adv = adv_attack.generate(inp, y_tgt)
                # get att
                att = scl_attack.generate(src, y, IndirectPGD, attack_args, y_tgt=self.target, mode=mode,
                                          nb_samples=nb_samples, verbose=False)
                # eval
                stats = e.eval(src, adv, att, y_src=y, y_adv=self.target)
                stats['Y'] = y.item()
                data.append(stats)

                if dump:
                    pickle.dump(data, open(fname, 'wb'))

        return data

    def get_acc(self, data: List, img_field: str = 'SRC', y_field: str = 'Y'):
        cnt = [stat['Y'] == scipy.stats.mode(stat[img_field][y_field])[0] for stat in data]
        acc = sum(cnt) / len(data) * 100
        return acc

    def collect(self, data: List, eps: int, tag: str, plot: bool = True):
        # Compute acc
        self.saved_results['ADV'].append((eps, self.get_acc(data, 'ADV', 'Y')))
        self.saved_results['ATT'].append((eps, self.get_acc(data, 'ATT', tag)))
        if plot:
            self.plot(tag)
        for f in ['ADV', 'ATT']:
            print(f, self.saved_results[f][-1][1])

    def plot(self, tt):
        plt.figure()
        for tag in ['ADV', 'ATT']:
            eps, acc = zip(*self.saved_results[tag])
            plt.plot(eps, acc, marker='o', markersize=3, label=tag.lower())
        plt.ylim(-5, 105)
        plt.yticks(list(range(0, 101, 10)))
        plt.legend()
        plt.savefig(f'test-{tt}-{eps[0]}.pdf')
        plt.close()


def plot_all():
    defenses = {'none': 'Y', 'median': 'Y_MED', 'random': 'Y_RND'}
    plt.figure(constrained_layout=True)

    for defense, tag in defenses.items():
        tester = AccBudget(class_net, args.target, lib='cv', algo='linear', tag='TEST')
        for eps in range(args.left, args.right, args.step):
            mode = 'cheap' if defense == 'random' else None
            d = tester.eval_ratio_eps(args.ratio, eps=eps, defense=defense, mode=mode, dump=False, load=True)
            if d is not None and len(d) == 100:
                tester.collect(d, eps, tag, plot=False)
        # plot adv
        if defense == 'none':
            eps, acc = zip(*tester.saved_results['ADV'])
            plt.plot(eps, acc, marker='o', markersize=3, label='PGD Attack')
        # plot att
        eps, acc = zip(*tester.saved_results['ATT'])
        plt.plot(eps, acc, marker='o', markersize=3, label=f'Scale-Adv Attack ({defense})')

    plt.ylim(-5, 65)
    plt.legend()
    plt.xticks(list(range(0, args.right + 1, 2)))
    plt.yticks(list(range(0, 61, 5)))
    plt.savefig('gen_acc-vs-L2.pdf')


if __name__ == '__main__':
    # set font for ccs
    mpl.rcParams['font.sans-serif'] = "Linux Libertine"
    mpl.rcParams['font.family'] = "sans-serif"
    mpl.rcParams['font.size'] = 15

    p = ArgumentParser()
    # Input args
    p.add_argument('--target', default=None, type=int)
    p.add_argument('--model', default=None, type=str, choices=ROBUST_MODELS)
    # Scaling args
    p.add_argument('--ratio', type=int, choices=(2, 3, 4, 5, 6))
    p.add_argument('--defense', type=str)
    # Attack args
    p.add_argument('-l', '--left', default=1, type=int)
    p.add_argument('-r', '--right', default=26, type=int)
    p.add_argument('-s', '--step', default=1, type=int)
    # Misc
    p.add_argument('--load', action='store_true')
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()

    # Load networks
    class_net = nn.Sequential(NormalizationLayer.from_preset('imagenet'), resnet50_imagenet(args.model)).eval()

    # plot all
    if args.plot:
        plot_all()
        exit()

    # attack
    tester = AccBudget(class_net, args.target, lib='cv', algo='linear', tag='TEST')
    for eps in range(args.left, args.right, args.step):
        mode = 'cheap' if args.defense == 'random' else None
        d = tester.eval_ratio_eps(args.ratio, eps=eps, defense=args.defense, mode=mode, dump=True, load=args.load)
        tag = {'random': 'Y_RND', 'median': 'Y_MED', 'none': 'Y'}
        tester.collect(d, eps, tag[args.defense])
