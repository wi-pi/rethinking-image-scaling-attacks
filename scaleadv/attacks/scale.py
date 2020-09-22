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

from scaleadv.datasets.imagenet import create_dataset


class ScaleAttack(object):

    def __init__(self, lib: SuppScalingLibraries, algo: SuppScalingAlgorithms, eps=1, bandwidth=2, allowed_changes=0.8,
                 verbose=False):
        assert ScalingGenerator.check_valid_lib_alg_input(lib, algo)
        self.lib, self.algo = lib, algo
        self.eps, self.bw, self.modi = eps, bandwidth, allowed_changes
        self.verbose = verbose
        self.collector = FourierPeakMatrixCollector(PeakMatrixMethod.optimization, self.algo, self.lib)
        self.scl_attack = QuadraticScaleAttack(eps=self.eps, verbose=self.verbose)
        self.scl_attack.optimize_runtime = True

    def generate(self, src: np.ndarray, tgt: np.ndarray):
        self._validate(src)
        self._validate(tgt)

        # scaling attack
        scl_method = ScalingGenerator.create_scaling_approach(src.shape, tgt.shape, self.lib, self.algo)
        scl, _, _ = self.scl_attack.attack(src, tgt, scl_method)

        # adaptive scaling attack
        defense = PreventionDefenseGenerator.create_prevention_defense(
            defense_type=PreventionTypeDefense.medianfiltering,
            scaler_approach=scl_method,
            fourierpeakmatrixcollector=self.collector,
            bandwidth=self.bw,
            verbose_flag=self.verbose,
            usecythonifavailable=True
        )
        adaptive_attack = AdaptiveAttackPreventionGenerator.create_adaptive_attack(
            defense_type=PreventionTypeDefense.medianfiltering,
            scaler_approach=scl_method,
            preventiondefense=defense,
            verbose_flag=self.verbose,
            usecythonifavailable=True,
            choose_only_unused_pixels_in_overlapping_case=False,
            allowed_changes=self.modi
        )
        ada = adaptive_attack.counter_attack(scl)

        return scl, ada, defense, scl_method

    def _validate(self, x: np.ndarray):
        assert x.dtype == np.uint8
        assert x.ndim == 3
        assert x.shape[-1] == 3


def test():
    # load data
    dataset = create_dataset(transform=None)
    _, src, y_src = dataset[1000]
    src = np.array(src, dtype=np.uint8)
    _, tgt, y_tgt = dataset[2000]
    tgt = np.array(tgt.resize((224, 224)), dtype=np.uint8)

    # load attacker
    lib = SuppScalingLibraries.PIL
    algo = SuppScalingAlgorithms.NEAREST
    attacker = ScaleAttack(lib, algo)

    # attack
    scl, ada, defense, _ = attacker.generate(src, tgt)

    # defense
    scl_def = defense.make_image_secure(scl)
    ada_def = defense.make_image_secure(ada)

    # save figs
    caps = 'src', 'tgt', 'scl', 'scl_def', 'ada', 'ada_def'
    for name in caps:
        var = locals()[name]
        Image.fromarray(var).save(f'{name}.png')


if __name__ == '__main__':
    test()
