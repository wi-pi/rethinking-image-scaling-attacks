### Defense 2, Adaptive Attack (Section 5.5)###
# We evaluate our defenses against an adaptive adversary who is aware of the defenses
# and adapts her attack accordingly.
import argparse
from itertools import product

from utils.plot_image_utils import plot_images_in_actual_size

from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingLibraries import SuppScalingLibraries
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.ScalingApproach import ScalingApproach
from attack.QuadrScaleAttack import QuadraticScaleAttack
from attack.ScaleAttackStrategy import ScaleAttackStrategy
from utils.load_image_data import load_image_examples

from defenses.prevention.PreventionDefenseType import PreventionTypeDefense
from defenses.prevention.PreventionDefenseGenerator import PreventionDefenseGenerator
from defenses.prevention.PreventionDefense import PreventionDefense

from defenses.detection.fourier.FourierPeakMatrixCollector import FourierPeakMatrixCollector, PeakMatrixMethod
from attack.adaptive_attack.AdaptiveAttackPreventionGenerator import AdaptiveAttackPreventionGenerator
from attack.adaptive_attack.AdaptiveAttack import AdaptiveAttackOnAttackImage

import time


usecythonifavailable = True
scaling_algorithm = SuppScalingAlgorithms.LINEAR
scaling_library = SuppScalingLibraries.CV
args_bandwidthfactor = 1.5
args_allowed_changes = 100 # the percentage of pixels that can be modified in each block.
eps_list = [0.1, 0.2, 0.4, 0.8, 1, 2, 4, 8, 16]


class Attack(object):

    def __init__(self, src_shape, tar_shape):
        """
        Args:
            src_shape: (H, W, C)
            tar_shape: (H, W, C)
        """
        self.scaler_approach = ScalingGenerator.create_scaling_approach(
            x_val_source_shape=src_shape,
            x_val_target_shape=tar_shape,
            lib=scaling_library,
            alg=scaling_algorithm
        )
        self.scale_att = QuadraticScaleAttack(eps=eps_list, verbose=True)
        self.scale_att.optimize_runtime = True
        self.fourierpeakmatrixcollector = FourierPeakMatrixCollector(
            method=PeakMatrixMethod.optimization,
            scale_library=scaling_library,
            scale_algorithm=scaling_algorithm
        )
        self.args_prevention_type = PreventionTypeDefense.medianfiltering
        self.preventiondefense = PreventionDefenseGenerator.create_prevention_defense(
            defense_type=self.args_prevention_type,
            scaler_approach=self.scaler_approach,
            fourierpeakmatrixcollector=self.fourierpeakmatrixcollector,
            bandwidth=args_bandwidthfactor,
            verbose_flag=True,
            usecythonifavailable=usecythonifavailable
        )
        self.adaptiveattack = AdaptiveAttackPreventionGenerator.create_adaptive_attack(
            defense_type=self.args_prevention_type,
            scaler_approach=self.scaler_approach,
            preventiondefense=self.preventiondefense,
            verbose_flag=True,
            usecythonifavailable=usecythonifavailable,
            choose_only_unused_pixels_in_overlapping_case=False,
            allowed_changes=args_allowed_changes/100
        )


    def attack(self, src, tar):
        """
        Args:
            src: (H, W, C) uint8 
            tar: (H, W, C) uint8
        """
        # Standard attack
        ts0 = time.time()
        attack_large, _, _ = self.scale_att.attack(
            src_image=src,
            target_image=tar,
            scaler_approach=self.scaler_approach
        )
        attack_large_scaled = self.scaler_approach.scale_image(xin=attack_large)

        # Adaptive attack
        ts1 = time.time()
        adaptive_large = self.adaptiveattack.counter_attack(att_image=attack_large)
        adaptive_large_scaled = self.preventiondefense.make_image_secure(att_image=adaptive_large)
        adaptive_large_scaled = self.scaler_approach.scale_image(xin=adaptive_large_scaled)

        ts2 = time.time()

        return attack_large, attack_large_scaled, adaptive_large, adaptive_large_scaled, ts1 - ts0, ts2 - ts1


from PIL import Image
import numpy as np
from typing import List
from pickle import load


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # setup
    root = '/u/g/y/gy/disk/Scale-Attack/Scale-Adv/static/bb_small/'
    # root = '/u/g/y/gy/disk/Scale-Attack/Scale-Adv/bb_test.'
    # budget = 5  # [budget]K
    id_list = range(0, 98, 2)
    budget_list = [1, 3, 7, 9]
    attack = Attack(src_shape=(672, 672, 3), tar_shape=(224, 224, 3))


    def get_src(i: int):
        pic = Image.open(f'{root}{i}.ratio_3.def_none.src_large.png')
        return np.array(pic)


    def get_tar(i: int, query: int):
        log = load(open(f'{root}{i}.ratio_3.def_none.log', 'rb'))

        pic = None
        for j, q, l2 in log:
            if q >= query:
                pic = Image.open(f'{root}{i}.ratio_3.def_none.{j:02d}.png')
                break

        return np.array(pic)


    def run(i, budget):
        src = get_src(i)
        tar = get_tar(i, budget * 1000)
        return src, tar, attack.attack(src, tar)


    def save(fig, name):
        Image.fromarray(fig).save(f'{root}{name}.png')


    for b, i in product(budget_list, id_list):
        try:
            print('START', b, i)
            src, tar, (att, att_down, ada, ada_down, ts1, ts2) = run(i, b)
        except Exception as err:
            print('ERROR', b, i, err)
        else:
            print('DONE', b, i, ts1, ts2)
            for k in 'src tar att att_down ada ada_down'.split():
                save(locals()[k], f'{i:02d}.{b}k.{k}')
