import os
import pickle

import numpy as np
from PIL import Image
from attack.QuadrScaleAttack import QuadraticScaleAttack
from defenses.detection.fourier.FourierPeakMatrixCollector import FourierPeakMatrixCollector, PeakMatrixMethod
from defenses.prevention.PreventionDefenseGenerator import PreventionDefenseGenerator
from defenses.prevention.PreventionDefenseType import PreventionTypeDefense
from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries
from torch.utils.data import DataLoader
from tqdm import tqdm

from scaleadv.datasets.imagenet import create_dataset

DATASET_PATH = 'static/datasets/imagenet-600'
OUTPUT_PATH = 'static/results/imagenet-600'
LIB = SuppScalingLibraries.PIL
ALGO = SuppScalingAlgorithms.NEAREST
TGT_SHAPE = (224, 224, 3)
FPM = 'fpm-cache-NEAREST.pkl'


def save(obj, name):
    Image.fromarray(obj).save(f'{OUTPUT_PATH}/{name}.png')


if __name__ == '__main__':
    # load data
    dataset = create_dataset(root=DATASET_PATH, transform=None)
    loader = DataLoader(dataset, batch_size=None, num_workers=8)

    # load static scaling tools
    if os.path.exists(FPM):
        fpm = pickle.load(open(FPM, 'rb'))
    else:
        fpm = FourierPeakMatrixCollector(PeakMatrixMethod.optimization, ALGO, LIB)
    scl_attack = QuadraticScaleAttack(eps=1, verbose=False)
    scl_attack.optimize_runtime = True

    # run through each image
    for index, x_img, _ in tqdm(loader):
        x_src = np.array(x_img)

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

        # get inp
        x_src_inp = scaling.scale_image(x_src)

        # get def_inp
        x_src_def = defense.make_image_secure(x_src)
        x_src_def_inp = scaling.scale_image(x_src_def)

        # save figs
        save(x_src, f'{index}.src')
        save(x_src_inp, f'{index}.src_inp')
        save(x_src_def, f'{index}.src_def')
        save(x_src_def_inp, f'{index}.src_def_inp')

        pickle.dump(fpm, open(FPM, 'wb'))
