import os
import pickle
import sys
from collections import OrderedDict
from typing import Tuple, Iterable

import numpy as np
import piq
import torch
import torch.nn as nn
from PIL import Image
from art.attacks import EvasionAttack
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier
from defenses.prevention.PreventionDefenseType import PreventionTypeDefense
from lpips import LPIPS
from prettytable import PrettyTable
from scaling.ScalingApproach import ScalingApproach
from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries
from torchvision.models import resnet50
from tqdm import tqdm

from scaleadv.attacks.adv import AdvAttack, _to_batch
from scaleadv.attacks.scale import ScaleAttack
from scaleadv.datasets.imagenet import create_dataset, IMAGENET_MEAN, IMAGENET_STD
from scaleadv.datasets.utils import ImageFolderWithIndex
from scaleadv.models.layers import NormalizationLayer


def show_table(data):
    data = dict(data.items())
    fields = ['y_pred', 'Linf', 'L2', 'PSNR', 'SSIM', 'MS-SSIM', 'LPIPS']
    t = PrettyTable([data.pop('y_true')] + fields)
    for k, v in data.items():
        t.add_row([k.replace('_def', '*')] + v[:1] + [f'{x:>6.3f}' for x in v[1:]])
    print(t)


class ScaleAdvAttack(object):

    def __init__(self, classifier: PyTorchClassifier, attacker: EvasionAttack, lib: SuppScalingLibraries,
                 algo: SuppScalingAlgorithms, input_shape: Tuple, up_factor: float = 1, save: str = None):
        self.AA = AdvAttack(classifier, attacker)
        self.SA = ScaleAttack(lib, algo, allowed_changes=0.7, bandwidth=2)
        self.lib, self.algo, self.input_shape = lib, algo, input_shape
        self.lpips = LPIPS(net='alex', verbose=False).cuda()
        self.save = save
        self.up_factor = up_factor

        if self.save:
            os.makedirs(self.save, exist_ok=True)

    @property
    def stats_file(self):
        return os.path.join(self.save, 'stats.pkl') if self.save else None

    def generate(self, dataset: ImageFolderWithIndex, indices: Iterable[int]):
        stats = OrderedDict()
        for i in tqdm(indices):
            try:
                stats[i] = self.generate_one(dataset, i)
                if self.stats_file:
                    pickle.dump(stats, open(self.stats_file, 'wb'))
                print(i)
                show_table(stats[i])
            except Exception as e:
                print(e)
        return stats

    def generate_one(self, dataset: ImageFolderWithIndex, index: int, large_inp: np.ndarray = None,
                     defense_type: PreventionTypeDefense = PreventionTypeDefense.medianfiltering,
                     get_defense: int = 0):
        # load data
        _, x_img, y_true = dataset[index]
        new_width = int(x_img.width * self.up_factor)
        new_height = int(x_img.height * self.up_factor)
        x_img = x_img.resize((new_width, new_height))
        x_src = np.array(x_img)

        # scale src to inp
        scaling = ScalingGenerator.create_scaling_approach(x_src.shape, self.input_shape, self.lib, self.algo)
        x_inp = scaling.scale_image(x_src)
        if large_inp is None:
            x_adv = self.AA.generate(x_inp)
        else:
            if large_inp.ndim == 4:
                x_being_attacked = np.stack(list(map(scaling.scale_image, large_inp)))
                x_adv = self.AA.generate(x_inp, x_being_attacked)
            else:
                x_being_attacked = scaling.scale_image(large_inp)
                x_adv = self.AA.generate(x_being_attacked)

        # process attack
        x_scl, x_ada, defense, scaling = self.SA.generate(src=x_src, tgt=x_adv, defense_type=defense_type)
        x_src_def = defense.make_image_secure(x_src)

        # generate full results
        stats = OrderedDict({'y_true': y_true})

        # small space
        for name in ['x_inp', 'x_adv']:
            var = locals()[name]
            stats[name] = self._evaluate(x_inp, var, None)
            self._save_fig(var, f'{index}.{name}')

        # large space
        for name in ['x_src', 'x_scl', 'x_ada']:
            # before defense
            var = locals()[name]
            stats[name] = self._evaluate(x_src, var, scaling)
            self._save_fig(var, f'{index}.{name}')

            # after defense
            var, name = defense.make_image_secure(var), f'{name}_def'
            stats[name] = self._evaluate(x_src, var, scaling)
            self._save_fig(var, f'{index}.{name}')

        # get src with defense
        if get_defense:
            x_src_def = []
            for _ in range(get_defense):
                x_src_def.append(defense.make_image_secure(x_src))
            x_src_def = np.stack(x_src_def)

        if get_defense:
            return stats, x_src_def
        return stats

    def _save_fig(self, x: np.ndarray, name: str):
        if self.save:
            assert x.dtype == np.uint8 and x.ndim == 3
            path = os.path.join(self.save, f'{name}.png')
            Image.fromarray(x).save(path)

    def _evaluate(self, anchor: np.ndarray, x: np.ndarray, scaling: ScalingApproach = None):
        x_inp = x if scaling is None else scaling.scale_image(x)
        y_pred, _ = self.AA.predict(x_inp)
        stats = [y_pred] + self._diff(anchor, x)
        return stats

    def _diff(self, x: np.ndarray, y: np.ndarray):
        x, y = map(lambda v: torch.as_tensor(_to_batch(v), dtype=torch.float32), [x, y])
        stats = [
            torch.norm(x - y, p=float('inf')),
            torch.norm(x - y, p=2),
            piq.psnr(x, y, data_range=1),
            piq.ssim(x, y, data_range=1),
            piq.multi_scale_ssim(x, y, data_range=1),
            self.lpips(x.cuda() * 2 - 1, y.cuda() * 2 - 1),
        ]
        stats = [i.squeeze().cpu().item() for i in stats]
        return stats


if __name__ == '__main__':
    # freq
    skip = int(sys.argv[1])

    # load AA
    model = nn.Sequential(
        NormalizationLayer(IMAGENET_MEAN, IMAGENET_STD),
        resnet50(pretrained=True)
    ).eval()
    classifier = PyTorchClassifier(model, nn.CrossEntropyLoss(), (3, 224, 224), 1000, clip_values=(0., 1.))
    attacker = ProjectedGradientDescentPyTorch(classifier, np.inf, 8 / 255., 1 / 255., max_iter=20)

    # load SA
    lib = SuppScalingLibraries.PIL
    algo = SuppScalingAlgorithms.NEAREST

    # load SAA
    path = 'static/results/test-saa/'
    os.makedirs(path, exist_ok=True)
    saa = ScaleAdvAttack(classifier, attacker, lib, algo, input_shape=(224, 224, 3), save=path)

    # test
    dataset = create_dataset(transform=None)
    indices = range(0, len(dataset), skip)
    stats = saa.generate(dataset, indices)

    # save stats
    filename = os.path.join(path, 'stats_final.pkl')
    pickle.dump(stats, open(filename, 'wb'))
