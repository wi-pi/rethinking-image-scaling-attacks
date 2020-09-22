import os
import pickle

import numpy as np
from PIL import Image
from attack.QuadrScaleAttack import QuadraticScaleAttack
from attack.adaptive_attack.AdaptiveAttackPreventionGenerator import AdaptiveAttackPreventionGenerator
from defenses.detection.fourier.FourierPeakMatrixCollector import FourierPeakMatrixCollector, PeakMatrixMethod
from defenses.prevention.PreventionDefenseGenerator import PreventionDefenseGenerator
from defenses.prevention.PreventionDefenseType import PreventionTypeDefense
from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries
from tqdm import tqdm

from scaleadv.datasets.imagenet import create_dataset

DATASET_PATH = 'static/datasets/imagenet-600'
OUTPUT_PATH = 'static/results/imagenet-600'
LIB = SuppScalingLibraries.PIL
ALGO = SuppScalingAlgorithms.NEAREST
TGT_SHAPE = (224, 224, 3)
FPM = 'fpm.pkl'


def load(name):
    img = Image.open(f'{OUTPUT_PATH}/{name}.png')
    return np.array(img)


def save(obj, name):
    Image.fromarray(obj).save(f'{OUTPUT_PATH}/{name}.png')


if __name__ == '__main__':
    # params
    NORM = 'inf'
    EPS = '2'
    ALLOWED_CHANGES = 0.8

    # load data
    dataset = create_dataset(root=DATASET_PATH, transform=None)

    # load static scaling tools
    fpm = pickle.load(open(FPM, 'rb'))
    scl_attack = QuadraticScaleAttack(eps=1, verbose=False)
    scl_attack.optimize_runtime = True

    # run through each image
    for index in tqdm(range(len(dataset))):
        # load all images needed
        src_name = f'{index}.src'
        src_adv_name = f'{index}.src_inp.adv_L{NORM}_{EPS}'
        def_adv_name = f'{index}.src_def_inp.adv_L{NORM}_{EPS}'
        x_src = load(src_name)
        x_src_inp_adv = load(src_adv_name)
        x_src_def_inp_adv = load(def_adv_name)

        # init scaling & defense
        scaling = ScalingGenerator.create_scaling_approach(x_src.shape, TGT_SHAPE, LIB, ALGO)
        defense = PreventionDefenseGenerator.create_prevention_defense(
            defense_type=PreventionTypeDefense.medianfiltering,
            scaler_approach=scaling,
            fourierpeakmatrixcollector=fpm,
            bandwidth=2,
            verbose_flag=False,
            usecythonifavailable=True
        )
        adaptive_attack = AdaptiveAttackPreventionGenerator.create_adaptive_attack(
            defense_type=PreventionTypeDefense.medianfiltering,
            scaler_approach=scaling,
            preventiondefense=defense,
            verbose_flag=False,
            usecythonifavailable=True,
            choose_only_unused_pixels_in_overlapping_case=False,
            allowed_changes=ALLOWED_CHANGES
        )

        # hide src_adv in src
        src_scl, _, _ = scl_attack.attack(x_src, x_src_inp_adv, scaling)
        src_scl_def = defense.make_image_secure(src_scl)
        src_ada = adaptive_attack.counter_attack(src_scl)
        src_ada_def = defense.make_image_secure(src_ada)

        # hide def_adv in src
        def_scl, _, _ = scl_attack.attack(x_src, x_src_def_inp_adv, scaling)
        def_scl_def = defense.make_image_secure(def_scl)
        def_ada = adaptive_attack.counter_attack(def_scl)
        def_ada_def = defense.make_image_secure(def_ada)

        # save figs (scl)
        save(src_scl, f'{src_adv_name}.scl')
        save(def_scl, f'{def_adv_name}.scl')

        # save figs (ada)
        save(src_ada, f'{src_adv_name}.ada')
        save(def_ada, f'{def_adv_name}.ada')

        # save figs (scl-def)
        save(src_scl_def, f'{src_adv_name}.scl_def')
        save(def_scl_def, f'{def_adv_name}.scl_def')

        # save figs (ada-def)
        save(src_ada_def, f'{src_adv_name}.ada_def')
        save(def_ada_def, f'{def_adv_name}.ada_def')
