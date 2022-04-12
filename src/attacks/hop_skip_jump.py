import types

import numpy as np
from art.attacks.evasion import HopSkipJump
from art.config import ART_NUMPY_DTYPE
from loguru import logger
from tqdm import trange

from src.attacks.smart_noise import SmartNoise


class HSJ(HopSkipJump):
    """Adapted from ART v1.10.0
    """

    def __init__(
        self,
        classifier,
        batch_size: int = 64,
        targeted: bool = False,
        norm: int | float | str = 2,
        max_iter: int = 50,
        max_eval: int = 10000,
        init_eval: int = 100,
        init_size: int = 100,
        verbose: bool = False,
        # New args
        max_query: int = 10000,
        smart_noise: SmartNoise | None = None,
    ):
        super().__init__(
            classifier=classifier, batch_size=batch_size, targeted=targeted, norm=norm,
            max_iter=max_iter, max_eval=max_eval, init_eval=init_eval, init_size=init_size, verbose=verbose
        )
        self.max_query = max_query
        self.smart_noise = smart_noise
        self.nb_query = 0
        self.log = []

        # patch classifier's prediction to record #query
        pred = self.estimator.predict

        def _pred(obj, x, *args, **kwargs):
            self.nb_query += x.shape[0]
            return pred(x, *args, **kwargs)

        setattr(self.estimator, 'predict', types.MethodType(_pred, self.estimator))

    def _attack(
        self,
        initial_sample: np.ndarray,
        original_sample: np.ndarray,
        target: int,
        mask: np.ndarray | None,
        clip_min: float,
        clip_max: float,
    ) -> np.ndarray:
        # Set current perturbed image to the initial image
        current_sample = initial_sample

        # Main loop to wander around the boundary
        for _ in (pbar := trange(self.max_iter, desc='HopSkipJump', bar_format='{l_bar}{r_bar}')):
            print()
            # First compute delta
            delta = self._compute_delta(
                current_sample=current_sample,
                original_sample=original_sample,
                clip_min=clip_min,
                clip_max=clip_max,
            )

            # Then run binary search
            current_sample = self._binary_search(
                current_sample=current_sample,
                original_sample=original_sample,
                norm=self.norm,
                target=target,
                clip_min=clip_min,
                clip_max=clip_max,
            )

            # Next compute the number of evaluations and compute the update
            num_eval = min(int(self.init_eval * np.sqrt(self.curr_iter + 1)), self.max_eval)

            update = self._compute_update(
                current_sample=current_sample,
                num_eval=num_eval,
                delta=delta,
                target=target,
                mask=mask,
                clip_min=clip_min,
                clip_max=clip_max,
            )

            # Finally run step size search by first computing epsilon
            if self.norm == 2:
                dist = np.linalg.norm(original_sample - current_sample)
            else:
                dist = np.max(abs(original_sample - current_sample))

            epsilon = 2.0 * dist / np.sqrt(self.curr_iter + 1)
            success = False

            while not success:
                epsilon /= 2.0
                potential_sample = current_sample + epsilon * update
                success = self._adversarial_satisfactory(  # type: ignore
                    samples=potential_sample[None],
                    target=target,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

            # Update current sample
            current_sample = np.clip(potential_sample, clip_min, clip_max)

            # Update current iteration
            self.curr_iter += 1

            # logging
            delta = original_sample - current_sample
            dist = np.linalg.norm(delta)
            self.log.append((self.nb_query, dist))
            pbar.set_postfix({'query': self.nb_query, 'l2': f'{dist:.5f}'})
            if self.nb_query > self.max_query:
                break

            # If attack failed. return original sample
            if np.isnan(current_sample).any():  # pragma: no cover
                logger.debug("NaN detected in sample, returning original sample.")
                return original_sample

        return current_sample

    def _compute_update(
        self,
        current_sample: np.ndarray,
        num_eval: int,
        delta: float,
        target: int,
        mask: np.ndarray | None,
        clip_min: float,
        clip_max: float,
    ) -> np.ndarray:
        rnd_noise_shape = [num_eval] + list(self.estimator.input_shape)
        if self.smart_noise:
            rnd_noise = self.smart_noise(current_sample, num_eval)
        else:
            if self.norm == 2:
                rnd_noise = np.random.randn(*rnd_noise_shape).astype(ART_NUMPY_DTYPE)
            else:
                rnd_noise = np.random.uniform(low=-1, high=1, size=rnd_noise_shape).astype(ART_NUMPY_DTYPE)

        # With mask
        if mask is not None:
            rnd_noise = rnd_noise * mask

        # Normalize random noise to fit into the range of input data
        rnd_noise = rnd_noise / np.sqrt(
            np.sum(
                rnd_noise ** 2,
                axis=tuple(range(len(rnd_noise_shape)))[1:],
                keepdims=True,
            )
        )
        eval_samples = np.clip(current_sample + delta * rnd_noise, clip_min, clip_max)
        rnd_noise = (eval_samples - current_sample) / delta

        # Compute gradient: This is a bit different from the original paper, instead we keep those that are
        # implemented in the original source code of the authors
        satisfied = self._adversarial_satisfactory(
            samples=eval_samples, target=target, clip_min=clip_min, clip_max=clip_max
        )
        f_val = 2 * satisfied.reshape([num_eval] + [1] * len(self.estimator.input_shape)) - 1.0
        f_val = f_val.astype(ART_NUMPY_DTYPE)

        if np.mean(f_val) == 1.0:
            grad = np.mean(rnd_noise, axis=0)
        elif np.mean(f_val) == -1.0:
            grad = -np.mean(rnd_noise, axis=0)
        else:
            f_val -= np.mean(f_val)
            grad = np.mean(f_val * rnd_noise, axis=0)

        # Compute update
        if self.norm == 2:
            result = grad / np.linalg.norm(grad)
        else:
            result = np.sign(grad)

        return result
