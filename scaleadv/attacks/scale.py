import numpy as np

from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingLibraries import SuppScalingLibraries
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from attack.QuadrScaleAttack import QuadraticScaleAttack
from attack.adaptive_attack.AdaptiveAttackPreventionGenerator import AdaptiveAttackPreventionGenerator
from defenses.detection.fourier.FourierPeakMatrixCollector import FourierPeakMatrixCollector, PeakMatrixMethod
from defenses.prevention.PreventionDefenseType import PreventionTypeDefense
from defenses.prevention.PreventionDefenseGenerator import PreventionDefenseGenerator

class ScaleAttack(object):

    def __init__(self, lib: SuppScalingLibraries, algo: SuppScalingAlgorithms, eps=1, bandwidth=2, allowed_changes=0.8, verbose=False):
        assert ScalingGenerator.check_valid_lib_alg_input(lib, algo)
        self.lib, self.algo = lib, algo
        self.eps, self.bw, self.modi = eps, bandwidth, allowed_changes
        self.verbose = verbose

    def generate(self, src: np.ndarray, tgt: np.ndarray):
        self._validate_input(src)
        self._validate_input(tgt)

        # scaling attack
        scale_approach = ScalingGenerator.create_scaling_approach(src.shape, tgt.shape, self.lib, self.algo)
        scale_attack = QuadraticScaleAttack(eps=self.eps, verbose=self.verbose)
        scale_result, _, _ = scale_attack.attack(src, tgt, scale_approach)

        # adaptive scaling attack
        defense = PreventionDefenseGenerator.create_prevention_defense(
            defense_type=PreventionTypeDefense.medianfiltering,
            scaler_approach=scale_approach,
            fourierpeakmatrixcollector=FourierPeakMatrixCollector(PeakMatrixMethod.optimization, self.algo, self.lib),
            bandwidth=self.bw,
            verbose_flag=self.verbose,
            usecythonifavailable=True
        )
        adaptive_attack = AdaptiveAttackPreventionGenerator.create_adaptive_attack(
            defense_type=PreventionTypeDefense.medianfiltering,
            scaler_approach=scale_approach,
            preventiondefense=defense,
            verbose_flag=self.verbose,
            usecythonifavailable=True,
            choose_only_unused_pixels_in_overlapping_case=False,
            allowed_changes=self.modi
        )
        scale_result_adaptive = adaptive_attack.counter_attack(scale_result)

        # apply median defense
        scale_result_protected = defense.make_image_secure(scale_result)
        scale_result_adaptive_protected = defense.make_image_secure(scale_result_adaptive)

        return scale_result, scale_result_protected, scale_result_adaptive, scale_result_adaptive_protected

    def _validate_input(self, x: np.ndarray):
        assert isinstance(x, np.ndarray)
        assert x.ndim == 3
        assert x.shape[-1] == 3
        assert x.min() >= 0.
        assert x.max() > 1
