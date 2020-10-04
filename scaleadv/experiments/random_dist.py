import concurrent.futures
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from attack.QuadrScaleAttack import QuadraticScaleAttack
from defenses.detection.fourier.FourierPeakMatrixCollector import FourierPeakMatrixCollector, PeakMatrixMethod
from defenses.prevention.PreventionDefenseGenerator import PreventionDefenseGenerator
from defenses.prevention.PreventionDefenseType import PreventionTypeDefense
from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries
from tqdm import tqdm

from scaleadv.datasets.imagenet import create_dataset

ID = 20000
LIB = SuppScalingLibraries.CV
ALGO = SuppScalingAlgorithms.CUBIC
TAG = 'cv_cubic'
OUTPUT = f'static/results/experiments/random_dist'
os.makedirs(OUTPUT, exist_ok=True)

if __name__ == '__main__':
    # load data
    dataset = create_dataset(transform=None)
    _, x_src, y_src = dataset[ID]
    x_src = np.array(x_src)

    # load SA
    fpm = FourierPeakMatrixCollector(PeakMatrixMethod.optimization, ALGO, LIB)
    attack = QuadraticScaleAttack(eps=1.0, verbose=False)
    scaling = ScalingGenerator.create_scaling_approach(x_src.shape, (224, 224, 3), LIB, ALGO)
    defense = PreventionDefenseGenerator.create_prevention_defense(
        defense_type=PreventionTypeDefense.randomfiltering,
        scaler_approach=scaling,
        fourierpeakmatrixcollector=fpm,
        bandwidth=2,
        verbose_flag=False,
        usecythonifavailable=True
    )

    # test
    x_inp = scaling.scale_image(x_src)


    def get_diff(_):
        x_def = defense.make_image_secure(x_src)
        x_def_inp = scaling.scale_image(x_def)
        diff = 0. + x_inp - x_def_inp
        return diff.copy()


    N = 140
    defense.make_image_secure(x_src)
    with concurrent.futures.ProcessPoolExecutor() as exe:
        output = list(tqdm(exe.map(get_diff, range(N)), total=N))
    data = np.concatenate(output)

    # plot
    sns.distplot(data)
    plt.savefig(f'{OUTPUT}/{ID}.{TAG}.pdf')
