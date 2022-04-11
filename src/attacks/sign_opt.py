import numpy as np
import torch
import torchvision.transforms.functional as F
from art.estimators.classification import PyTorchClassifier
from numpy import linalg as LA
from tqdm import trange

from src.attacks.smart_noise import SmartNoise


class QueryLimitSignOPT(object):

    def __init__(
        self,
        model: PyTorchClassifier,
        max_iter: int = 1000,
        num_eval: int = 200,
        alpha: float = 0.2,
        beta: float = 0.001,
        # New args
        max_query: int = 10000,
        smart_noise: SmartNoise | None = None,
    ):
        self.model = model
        self.max_iter = max_iter
        self.num_eval = num_eval
        self.alpha = alpha
        self.beta = beta

        self.max_query = max_query
        self.smart_noise = smart_noise
        self.log = []

    def is_adversary(self, x: np.ndarray, y: int):
        x = np.clip(x, 0, 1)
        assert x.dtype == np.float32
        pred = self.model.predict(x).argmax(1).item()
        return pred != y

    def generate(self, x0: np.ndarray, y0: int, tag=None):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """
        # type check
        assert x0.shape[0] == 1
        assert x0.dtype == np.float32

        if self.is_adversary(x0, y0):
            print("Fail to classify the image. No need to attack.")
            return x0, 0, True, 0, None

        """
        Init: params
        """
        alpha = self.alpha
        beta = self.beta
        nb_query = 0

        """
        Init: find a good starting point (direction)
        """
        best_theta, g_theta = None, float('inf')
        for _ in range(100):
            # randomly sample theta from gaussian distribution
            theta = np.random.randn(*x0.shape).astype(np.float32)

            # check if theta is an adv example
            nb_query += 1
            if self.is_adversary(x0 + theta, y0):
                # l2 normalize
                initial_lbd = LA.norm(theta)
                theta = theta / initial_lbd

                # binary search a point near the boundary
                lbd, count = self.fine_grained_binary_search(x0, y0, theta, initial_lbd, g_theta)
                nb_query += count

                # record the best theta
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd

        # cannot find an adv example
        if g_theta == float('inf'):
            print("Couldn't find valid initial, failed")
            return x0, 0, False, nb_query, best_theta

        print(f"========> Found best distortion {g_theta:.4f} using {nb_query} queries")

        """
        Gradient Descent
        """
        xg, gg = best_theta, g_theta
        vg = np.zeros_like(xg)
        with trange(self.max_iter, desc='SignOPT') as pbar:
            for i in pbar:
                # estimate the gradient at x0 + theta
                sign_gradient, grad_queries = self.sign_grad_v1(x0, y0, xg, initial_lbd=gg, h=beta)

                # line search of the step size of gradient descent
                ls_count = 0
                min_theta = xg  # next theta
                min_g2 = gg  # current g_theta
                min_vg = vg  # velocity (for momentum only)
                for _ in range(15):
                    # update theta by one-step sgd
                    new_theta = xg - alpha * sign_gradient
                    new_theta = new_theta / LA.norm(new_theta)

                    new_g2, count = self.fine_grained_binary_search_local(x0, y0, new_theta, initial_lbd=min_g2,
                                                                          tol=beta / 500)
                    ls_count += count
                    alpha = alpha * 2  # gradually increasing step size
                    if new_g2 < min_g2:
                        min_theta = new_theta
                        min_g2 = new_g2
                    else:
                        break

                # if the above code failed for the init alpha, we then try to decrease alpha
                if min_g2 >= gg:
                    for _ in range(15):
                        alpha = alpha * 0.25
                        new_theta = xg - alpha * sign_gradient
                        new_theta = new_theta / LA.norm(new_theta)
                        new_g2, count = self.fine_grained_binary_search_local(x0, y0, new_theta, initial_lbd=min_g2,
                                                                              tol=beta / 500)
                        ls_count += count
                        if new_g2 < gg:
                            min_theta = new_theta
                            min_g2 = new_g2
                            break

                # if the above two blocks of code failed
                if alpha < 1e-4:
                    alpha = 1.0
                    print("Warning: not moving")
                    beta = beta * 0.1
                    if beta < 1e-8:
                        break

                # if all attempts failed, min_theta, min_g2 will be the current theta (i.e. not moving)
                xg, gg = min_theta, min_g2
                vg = min_vg

                # save image
                if tag is not None:
                    adv = np.clip(x0 + gg * xg, 0, 1)[0]
                    F.to_pil_image(torch.as_tensor(adv)).save(f'{tag}.{i:02d}.png')

                # logging
                nb_query += grad_queries + ls_count
                pbar.set_postfix({'query': nb_query, 'l2': f'{gg:.3f}'})
                self.log.append((i, nb_query, gg))

                # stop if we reach the query limit
                if nb_query > self.max_query:
                    break

        return x0 + gg * xg, gg, True, nb_query, xg

    def sign_grad_v1(self, x0: np.ndarray, y0: int, theta: np.ndarray, initial_lbd: float, h: str = 0.001):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        sign_grad = np.zeros_like(theta)
        x_adv = x0 + initial_lbd * theta

        if self.smart_noise:
            u_all = self.smart_noise(x_adv, self.num_eval)
        else:
            u_all = np.random.randn(self.num_eval, *theta.shape).astype(np.float32)

        for i in range(self.num_eval):
            # get unit noise
            u = u_all[i]
            u = u / LA.norm(u)

            # get unit new theta
            new_theta = theta + h * u
            new_theta = new_theta / LA.norm(new_theta)

            sign = -1 if self.is_adversary(x0 + initial_lbd * new_theta, y0) else 1
            sign_grad += u * sign

        sign_grad /= self.num_eval

        return sign_grad, self.num_eval

    def fine_grained_binary_search_local(
        self, x0: np.ndarray, y0: int, theta: np.ndarray,
        initial_lbd: float = 1.0, tol: str = 1e-5
    ):
        nb_queries = 0
        lbd = initial_lbd

        # still inside boundary
        nb_queries += 1
        if not self.is_adversary(x0 + lbd * theta, y0):
            lbd_lo, lbd_hi = lbd, lbd * 1.01
            while not self.is_adversary(x0 + lbd_hi * theta, y0):
                lbd_hi = lbd_hi * 1.01
                nb_queries += 1
                if lbd_hi > 20:
                    return float('inf'), nb_queries
        else:
            lbd_lo, lbd_hi = lbd * 0.99, lbd
            while self.is_adversary(x0 + lbd_lo * theta, y0):
                lbd_lo = lbd_lo * 0.99
                nb_queries += 1

        while lbd_hi - lbd_lo > tol:
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nb_queries += 1
            if self.is_adversary(x0 + lbd_mid * theta, y0):
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid

        return lbd_hi, nb_queries

    def fine_grained_binary_search(
        self, x0: np.ndarray, y0: int, theta: np.ndarray,
        initial_lbd: float, current_best: float
    ):
        nb_queries = 0
        if initial_lbd > current_best:
            nb_queries += 1
            if not self.is_adversary(x0 + current_best * theta, y0):
                return float('inf'), nb_queries
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_lo, lbd_hi = 0.0, lbd
        while lbd_hi - lbd_lo > 1e-3:  # was 1e-5
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nb_queries += 1
            if self.is_adversary(x0 + lbd_mid * theta, y0):
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid

        return lbd_hi, nb_queries
