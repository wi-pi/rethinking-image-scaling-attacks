import numpy as np
import torch
from art.estimators.classification import PyTorchClassifier
from numpy import linalg as LA

# from qpsolvers import solve_qp

MAX_ITER = 1000


class SignOPT_ModelAdaptor(object):

    def __init__(self, model: PyTorchClassifier):
        self.model = model

    def predict_label(self, x: torch.Tensor):
        x = x.clamp(0, 1)
        p = self.model.predict(x.cpu().numpy()).argmax(1)
        return p.item()


class SignOPT(object):
    def __init__(self, model, k=200, preprocess=None, smart_noise=False):
        self.model = SignOPT_ModelAdaptor(model)
        self.k = k
        self.log = torch.ones(MAX_ITER, 2)
        self.preprocess = preprocess
        self.smart_noise = smart_noise

    def generate(self, x0: np.ndarray, y0: int, alpha=0.2, beta=0.001, iterations=1000, query_limit=20000):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """

        query_count = 0
        ls_total = 0
        x0 = torch.as_tensor(x0)

        if self.model.predict_label(x0) != y0:
            print("Fail to classify the image. No need to attack.")
            return x0, 0, True, 0, None

        """
        Init: find a good starting point (direction)
        """
        num_directions = 100
        best_theta, g_theta = None, float('inf')
        for _ in range(num_directions):
            # randomly sample theta from gaussian distribution
            theta = torch.randn_like(x0)

            # check if theta is an adv example
            pred = self.model.predict_label(x0 + theta)
            query_count += 1

            # register adv directions
            if pred != y0:
                # l2 normalize
                initial_lbd = LA.norm(theta)
                theta /= initial_lbd

                # binary search a point near the boundary
                lbd, count = self.fine_grained_binary_search(self.model, x0, y0, theta, initial_lbd, g_theta)
                query_count += count

                # record the best theta
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    # print("--------> Found distortion %.4f" % g_theta)

        # cannot find an adv example
        if g_theta == float('inf'):
            print("Couldn't find valid initial, failed")
            return x0, 0, False, query_count, best_theta

        # print(f"========> Found best distortion {g_theta:.4f} using {query_count} queries")
        self.log[0][0], self.log[0][1] = g_theta, query_count

        """
        Gradient Descent
        """
        xg, gg = best_theta, g_theta
        vg = np.zeros_like(xg)
        distortions = [gg]
        for i in range(iterations):
            # estimate the gradint at x0 + theta
            sign_gradient, grad_queries = self.sign_grad_v1(x0, y0, xg, initial_lbd=gg, h=beta)

            # line search of the step size of gradient descent
            ls_count = 0
            min_theta = xg  # next theta
            min_g2 = gg  # current g_theta
            min_vg = vg  # velocity (for momentum only)
            for _ in range(15):
                # update theta by one-step sgd
                new_theta = xg - alpha * sign_gradient
                new_theta /= LA.norm(new_theta)

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
                    new_theta /= LA.norm(new_theta)
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

            # update logs
            query_count += grad_queries + ls_count
            ls_total += ls_count
            distortions.append(gg)

            # stop if we reach the query limit
            if query_count > query_limit:
                break

            # logging
            self.log[i + 1][0], self.log[i + 1][1] = gg, query_count
            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1:3d} distortion {gg:.4f} num_queries {query_count}")

        target = self.model.predict_label(x0 + gg * xg)
        print(f"Succeed distortion {gg:.4f} target {target:d} queries {query_count:d} LS queries {ls_total:d}")
        return x0 + gg * xg, gg, True, query_count, xg

    def sign_grad_v1(self, x0, y0, theta, initial_lbd, h=0.001):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        sign_grad = torch.zeros_like(theta)
        queries = 0
        for i in range(self.k):
            if self.smart_noise and self.preprocess is not None:
                u = self.get_noise(x0)
            else:
                u = torch.randn_like(theta)

            u /= u.norm(2)
            new_theta = theta + h * u
            new_theta /= new_theta.norm(2)

            queries += 1
            pred = self.model.predict_label(x0 + initial_lbd * new_theta)
            sign = 1 if pred == y0 else -1
            sign_grad += u * sign

        sign_grad /= self.k

        return sign_grad, queries

    def fine_grained_binary_search_local(self, x0, y0, theta, initial_lbd=1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd

        # still inside boundary
        if self.model.predict_label(x0 + lbd * theta) == y0:
            lbd_lo = lbd
            lbd_hi = lbd * 1.01
            nquery += 1
            while self.model.predict_label(x0 + lbd_hi * theta) == y0:
                lbd_hi = lbd_hi * 1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd * 0.99
            nquery += 1
            while self.model.predict_label(x0 + lbd_lo * theta) != y0:
                lbd_lo = lbd_lo * 0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if self.model.predict_label(x0 + lbd_mid * theta) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def fine_grained_binary_search(self, model, x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best:
            if model.predict_label(x0 + current_best * theta) == y0:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_hi = lbd
        lbd_lo = 0.0

        while (lbd_hi - lbd_lo) > 1e-3:  # was 1e-5
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if model.predict_label(x0 + lbd_mid * theta) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def get_noise(self, x0: torch.Tensor):
        x_hr = x0.clone().cuda()
        delta_lr = torch.randn(1, 3, 224, 224).cuda()
        delta_hr = torch.zeros_like(x_hr).requires_grad_()
        perturbed_hr = x_hr + delta_hr
        perturbed_lr = self.preprocess(x_hr) + delta_lr
        loss = torch.norm(self.preprocess(perturbed_hr) - perturbed_lr)
        loss.backward()
        return delta_hr.grad.cpu()
