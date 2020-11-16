"""
This module conducts experiments to compare an adversary's incentive regarding
1. PSNR of adv-example with budget eps.
2. PSNR of attack example generated from adv-example in 1.
"""
import os
import pickle
from typing import List

import matplotlib.pyplot as plt

from scaleadv.tests.scale_adv import *


class MetricCompare(object):
    # PGD params
    NORM = 2
    STEP = 30

    # Scale-Adv Params
    # LR = 0.01
    # ITER = 1000
    # LAM_INP = 7

    # Scale-Adv Params (random)
    LR = 0.1
    ITER = 120
    LAM_INP = 100

    # PIL array to batch array
    pil_to_batch = T.Compose([T.ToTensor(), lambda x: x.numpy()[None, ...]])

    def __init__(self, class_net: nn.Module, src: np.ndarray, target: int, eps: List[int], tag: str):
        assert src.ndim == 3 and src.shape[-1] == 3
        self.class_net = class_net
        self.src = src
        self.target = target
        self.eps = eps
        self.tag = tag

    def eval_lib_algo(self, lib: str, algo: str, defense: str = None, dump=True, load=False):
        fname = f'{self.tag}.{lib}.{algo}.{defense}.pkl'
        if load and os.path.exists(fname):
            return pickle.load(open(fname, 'rb'))

        # Load scaling algorithm
        lib_t, algo_t = LIB_TYPE[lib], ALGO_TYPE[algo]
        scaling = ScalingGenerator.create_scaling_approach(self.src.shape, INPUT_SHAPE_PIL, lib_t, algo_t)
        mask = get_mask_from_cl_cr(scaling.cl_matrix, scaling.cr_matrix)

        # Get src and inp as batch ndarray
        inp = scaling.scale_image(self.src)
        src, inp = map(self.pil_to_batch, [self.src, inp])

        # Get pooling layer
        pooling = NonePool2d()
        kernel = src.shape[2] // inp.shape[2] * 2 - 1
        pooling_args = kernel, 1, kernel // 2, mask
        nb_samples = NUM_SAMPLES_SAMPLE if defense in ['random', 'laplace'] else 1
        if defense not in [None, 'none']:
            pooling = POOLING[defense](*pooling_args)

        # Get networks
        scale_net = ScaleNet(scaling.cl_matrix, scaling.cr_matrix).eval()
        class_net = self.class_net.eval()
        if nb_samples > 1:
            class_net = BalancedDataParallel(FIRST_GPU_BATCH, class_net)
        scale_net = scale_net.cuda()
        class_net = class_net.cuda()

        # Load art's classifier
        y = np.eye(NUM_CLASSES, dtype=int)[None, self.target]
        classifier = PyTorchClassifier(class_net, nn.CrossEntropyLoss(), INPUT_SHAPE_NP, NUM_CLASSES,
                                       clip_values=(0, 1))

        # Load scaling attack
        scl_attack = ScaleAttack(scale_net, class_net, pooling, lr=self.LR, max_iter=self.ITER, lam_inp=self.LAM_INP,
                                 nb_samples=nb_samples, early_stop=True)
        e = Evaluator(scale_net, class_net, pooling_args)

        data = []
        for eps in self.eps:
            # Get adv
            eps_step = 2.5 * eps / self.STEP
            adv_attack = IndirectPGD(classifier, self.NORM, eps, eps_step, self.STEP, targeted=True)
            adv = adv_attack.generate(inp, y)

            # Get att
            adaptive = defense != 'none'
            att = scl_attack.generate(src, adv, adaptive, 'sample', y_tgt=self.target)

            # Test
            stats = e.eval(src, adv, att, summary=True, tag=f'EVAL.{lib}.{algo}.{defense}.eps{eps}', save='.',
                           y_adv=self.target)
            data.append(stats)

        if dump:
            pickle.dump(data, open(fname, 'wb'))
        return data

    def plot_lib_algo(self, lib: str, algo: str, defense: str, marker: str):
        data = self.eval_lib_algo(lib, algo, defense, dump=True, load=True)
        y_label = {'none': 'Y', 'median': 'Y_MED', 'random': 'Y_RND'}[defense]
        x, y = [], []
        warning = []
        for d in data:
            adv_ok = np.mean(d['ADV']['Y'] == args.target) > 0.7
            att_ok = np.mean(d['ATT'][y_label] == args.target) > 0.7
            adv_score = d['ADV']['L-2'].cpu().item()
            att_score = d['ATT']['L-2'].cpu().item() / 3 if adv_ok else adv_score
            if adv_ok:
                x.append(adv_score)
                y.append(att_score)
            if adv_ok and not att_ok:
                warning.append((adv_score, att_score))

        if x:
            plt.plot(x, y, marker, lw=1, ms=1.75, label=f'{lib}.{algo} ({defense})')
            # plt.plot(x, y, marker, label=f'{lib}.{algo} ({defense})')
        if warning:
            plt.scatter(*zip(*warning), s=2, c='k', marker='o')


if __name__ == '__main__':
    p = ArgumentParser()
    # Input args
    p.add_argument('--id', type=int, required=True, help='ID of test image')
    p.add_argument('--target', type=int, required=True, help='target label')
    p.add_argument('--robust', default=None, type=str, choices=ROBUST_MODELS, help='use robust model, optional')
    # Scaling args
    p.add_argument('--bigger', default=1, type=int, help='scale up the source image')
    # Adversarial attack args
    p.add_argument('--eps', default=20, type=float, help='L2 perturbation of adv-example')
    p.add_argument('--step', default=30, type=int, help='max iterations of PGD attack')
    p.add_argument('--adv-proxy', action='store_true', help='do adv-attack on noisy proxy')
    # Scaling attack args
    p.add_argument('--lr', default=0.01, type=float, help='learning rate for scaling attack')
    p.add_argument('--lam-inp', default=1, type=int, help='lambda for L2 penalty at the input space')
    p.add_argument('--lam-ce', default=2, type=int, help='lambda for CE penalty')
    p.add_argument('--iter', default=200, type=int, help='max iterations of Scaling attack')
    p.add_argument('--defense', default=None, type=str, choices=POOLING.keys(), help='type of defense')
    p.add_argument('--mode', default='none', type=str, choices=ADAPTIVE_MODE, help='adaptive attack mode')
    args = p.parse_args()

    # Load data
    dataset = create_dataset(transform=None)
    src, _ = dataset[args.id]
    src = resize_to_224x(src, more=args.bigger)
    src = np.array(src)
    ratio = src.shape[0] // 224

    # Load networks
    class_net = nn.Sequential(NormalizationLayer.from_preset('imagenet'), resnet50_imagenet(args.robust)).eval()

    # Other params
    eps = list(range(5, 51, 5))
    lib = 'cv'
    algo_list = ['linear', 'area']
    defense_list = ['none', 'median', 'cheap']
    color = 'o- o--'.split()
    marker = 'C2 C0 C1'.split()

    # Tester
    tester = MetricCompare(class_net, src, args.target, eps, tag=f'eval.{args.id}')

    # Test + Plot
    plt.figure(figsize=(5, 5), constrained_layout=True)
    plt.title(f'Scale-Adv Attack')
    plt.xlabel('Adversarial Attack (L-2)')
    plt.ylabel('Scale-Adv Attack (L-2)')
    for algo, c in zip(algo_list, color):
        for defense, m in zip(defense_list, marker):
            tester.plot_lib_algo(lib, algo, defense, marker=f'{m}{c}')

    # Plot reference line
    eps[0] = 10
    plt.plot([eps[0], eps[-1]], [eps[0], eps[-1]], 'k--', lw=1, label='reference')
    plt.legend()
    plt.savefig(f'test.pdf')
