import time

import numpy as np
import torch
from numpy import linalg as LA
from tqdm import trange

from scaleadv.attacks.utils import SignOPT_ModelAdaptor, improve_theta, inverse_preprocess_np

# from qpsolvers import solve_qp

start_learning_rate = 1.0
MAX_ITER = 1000
sp = [1, 3, 224, 224]
# sp = [1, 3, 224*3, 224*3]


def sign(y):
    """
    y -- numpy array of shape (m,)
    Returns an element-wise indication of the sign of a number.
    The sign function returns -1 if y < 0, 1 if x >= 0. nan is returned for nan inputs.
    """
    y_sign = np.sign(y)
    y_sign[y_sign == 0] = 1
    return y_sign


class OPT_attack_sign_SGD(object):
    def __init__(self, model, k=200, train_dataset=None, preprocess=None):
        self.model = SignOPT_ModelAdaptor(model)
        self.k = k
        self.train_dataset = train_dataset
        self.log = torch.ones(MAX_ITER, 2)
        self.preprocess = preprocess

    def get_log(self):
        return self.log

    def attack_untargeted(self, x0, y0, alpha=0.2, beta=0.001, iterations=1000, query_limit=20000,
                          distortion=None, svm=False, momentum=0.0, stopping=0.0001):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """
        np.random.seed(0)

        model = self.model
        y0 = y0[0]
        query_count = 0
        ls_total = 0

        if (model.predict_label(x0) != y0):
            print("Fail to classify the image. No need to attack.")
            return x0, 0, True, 0, None

        #### init: Calculate a good starting point (direction)
        num_directions = 100
        best_theta, g_theta = None, float('inf')
        print("Searching for the initial direction on %d random directions: " % (num_directions))
        timestart = time.time()
        for i in trange(num_directions):
            query_count += 1
            theta = np.random.randn(*sp)  # gaussian distortion
            theta = inverse_preprocess_np(self.preprocess, x0, theta)
            # register adv directions
            if model.predict_label(x0 + torch.tensor(theta, dtype=torch.float).cuda()) != y0:
                initial_lbd = LA.norm(theta)
                theta /= initial_lbd  # l2 normalize
                lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    print("--------> Found distortion %.4f" % g_theta)
        timeend = time.time()

        ## fail if cannot find a adv direction within 200 Gaussian
        if g_theta == float('inf'):
            print("Couldn't find valid initial, failed")
            return x0, 0, False, query_count, best_theta
        print("==========> Found best distortion %.4f in %.4f seconds "
              "using %d queries" % (g_theta, timeend - timestart, query_count))
        self.log[0][0], self.log[0][1] = g_theta, query_count

        #### Begin Gradient Descent.
        timestart = time.time()
        xg, gg = best_theta, g_theta
        vg = np.zeros_like(xg)
        learning_rate = start_learning_rate
        prev_obj = 100000
        distortions = [gg]
        cc = 0
        for i in range(iterations):
            # # Improve theta below...
            # if self.preprocess and i % 5 == 0:
            #     xg_new, gg_new = improve_theta(self.preprocess, x0, xg, gg)
            #     gg_new /= 10
            #     gp, count = self.fine_grained_binary_search_local(model, x0, y0, xg_new * gg_new, initial_lbd=1.0,
            #                                                       tol=beta / 500)
            #     gg_new *= gp
            #     cc += count
            #     print(f'Improve using {cc} queries, from {gg:.4f} to {gg_new:.4f}')
            #     if gg_new < gg:
            #         xg, gg = xg_new, gg_new

            ## gradient estimation at x0 + theta (init)
            if svm == True:
                sign_gradient, grad_queries = self.sign_grad_svm(x0, y0, xg, initial_lbd=gg, h=beta)
            else:
                sign_gradient, grad_queries = self.sign_grad_v1(x0, y0, xg, initial_lbd=gg, h=beta)

            ## Line search of the step size of gradient descent
            ls_count = 0
            min_theta = xg  ## next theta
            min_g2 = gg  ## current g_theta
            min_vg = vg  ## velocity (for momentum only)
            for _ in range(15):
                # update theta by one step sgd
                if momentum > 0:
                    new_vg = momentum * vg - alpha * sign_gradient
                    new_theta = xg + new_vg
                else:
                    new_theta = xg - alpha * sign_gradient
                new_theta /= LA.norm(new_theta)

                new_g2, count = self.fine_grained_binary_search_local(
                    model, x0, y0, new_theta, initial_lbd=min_g2, tol=beta / 500)
                ls_count += count
                alpha = alpha * 2  # gradually increasing step size
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                    if momentum > 0:
                        min_vg = new_vg
                else:
                    break

            if min_g2 >= gg:  ## if the above code failed for the init alpha, we then try to decrease alpha
                for _ in range(15):
                    alpha = alpha * 0.25
                    if momentum > 0:
                        new_vg = momentum * vg - alpha * sign_gradient
                        new_theta = xg + new_vg
                    else:
                        new_theta = xg - alpha * sign_gradient
                    new_theta /= LA.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local(
                        model, x0, y0, new_theta, initial_lbd=min_g2, tol=beta / 500)
                    ls_count += count
                    if new_g2 < gg:
                        min_theta = new_theta
                        min_g2 = new_g2
                        if momentum > 0:
                            min_vg = new_vg
                        break

            if alpha < 1e-4:  ## if the above two blocks of code failed
                alpha = 1.0
                print("Warning: not moving")
                beta = beta * 0.1
                if (beta < 1e-8):
                    break

            ## if all attemps failed, min_theta, min_g2 will be the current theta (i.e. not moving)
            xg, gg = min_theta, min_g2
            vg = min_vg

            query_count += (grad_queries + ls_count)
            ls_total += ls_count
            distortions.append(gg)

            if query_count > query_limit:
                break

            ## logging
            if (i + 1) % 1 == 0:
                p = model.predict_label(x0 + torch.tensor(gg * xg, dtype=torch.float).cuda()).cpu().item()
                print("Iteration %3d distortion %.4f num_queries %d pred %d grad %.3f" % (i + 1, gg, query_count, p, LA.norm(sign_gradient)))
            self.log[i + 1][0], self.log[i + 1][1] = gg, query_count
            # if distortion is not None and gg < distortion:
            #    print("Success: required distortion reached")
            #    break

        if distortion is None or gg < distortion:
            target = model.predict_label(x0 + torch.tensor(gg * xg, dtype=torch.float).cuda()).cpu().item()
            print("Succeed distortion {:.4f} target {:d} queries {:d} LS queries {:d}\n".format(gg, target, query_count,
                                                                                                ls_total))
            return x0 + torch.tensor(gg * xg, dtype=torch.float).cuda(), gg, True, query_count, xg

        timeend = time.time()
        print("\nFailed: distortion %.4f" % (gg))

        self.log[i + 1:, 0] = gg
        self.log[i + 1:, 1] = query_count
        return x0 + torch.tensor(gg * xg, dtype=torch.float).cuda(), gg, False, query_count, xg

    def sign_grad_fast(self, x0, y0, theta, initial_lbd, h=0.001, D=4, target=None):
        # sample K Gaussian noise
        k = self.k
        u = torch.randn(k, *theta.shape[1:]).cuda()
        u /= u.view(k, -1).norm(dim=1)[:, None, None, None]

        # compute K new theta's
        new_theta = torch.tensor(theta, dtype=torch.float).cuda() + h * u
        new_theta /= new_theta.view(k, -1).norm(dim=1)[:, None, None, None]

        # predict
        pred = self.model.predict_label(x0 + initial_lbd * new_theta)
        pred = pred != y0 if target is None else pred == target

        # estimate sign grad
        sign = (1 - 2 * pred)[:, None, None, None]  # (0, 1) -> (+1, -1)
        sign_grad = torch.mean(u * sign, dim=0, keepdims=True)

        return sign_grad.cpu().numpy(), k

    def sign_grad_v1(self, x0, y0, theta, initial_lbd, h=0.001, D=4, target=None):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        K = self.k  # 200 random directions (for estimating the gradient)
        sign_grad = np.zeros(theta.shape)
        queries = 0

        xb = x0 + torch.tensor(initial_lbd * theta, dtype=torch.float).cuda()
        # from IPython import embed; embed(using=False); exit()

        for iii in trange(K):  # for each u
            u = np.random.randn(*sp); u /= LA.norm(u)
            u = inverse_preprocess_np(self.preprocess, xb, u)
            hu = u / LA.norm(u) * h
            new_theta = theta + hu; new_theta /= LA.norm(new_theta)
            sign = 1

            # Targeted case.
            if (target is not None and
                    self.model.predict_label(
                        x0 + torch.tensor(initial_lbd * new_theta, dtype=torch.float).cuda()) == target):
                sign = -1

            # Untargeted case
            # preds.append(self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).cuda()).item())
            if (target is None and
                    self.model.predict_label(
                        x0 + torch.tensor(initial_lbd * new_theta, dtype=torch.float).cuda()) != y0):  # success
                sign = -1

            queries += 1
            # sign_grad += u * sign
            sign_grad += hu / LA.norm(hu) * sign

        sign_grad /= K

        # sign_grad_u = sign_grad/LA.norm(sign_grad)
        # new_theta = theta + h*sign_grad_u
        # new_theta /= LA.norm(new_theta)
        # fxph, q1 = self.fine_grained_binary_search_local(self.model, x0, y0, new_theta, initial_lbd=initial_lbd, tol=h/500)
        # delta = (fxph - initial_lbd)/h
        # queries += q1
        # sign_grad *= 0.5*delta

        return sign_grad, queries

    ##########################################################################################
    def sign_grad_v2(self, x0, y0, theta, initial_lbd, h=0.001, K=200):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        sign_grad = np.zeros(theta.shape)
        queries = 0
        for _ in range(K):
            u = np.random.randn(*theta.shape)
            u /= LA.norm(u)

            ss = -1
            new_theta = theta + h * u
            new_theta /= LA.norm(new_theta)
            if self.model.predict_label(x0 + torch.tensor(initial_lbd * new_theta, dtype=torch.float).cuda()) == y0:
                ss = 1
            queries += 1
            sign_grad += sign(u) * ss
        sign_grad /= K
        return sign_grad, queries

    def sign_grad_svm(self, x0, y0, theta, initial_lbd, h=0.001, K=100, lr=5.0, target=None):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        sign_grad = np.zeros(theta.shape)
        queries = 0
        dim = np.prod(theta.shape)
        X = np.zeros((dim, K))
        for iii in range(K):
            u = np.random.randn(*theta.shape)
            u /= LA.norm(u)

            sign = 1
            new_theta = theta + h * u
            new_theta /= LA.norm(new_theta)

            # Targeted case.
            if (target is not None and
                    self.model.predict_label(
                        x0 + torch.tensor(initial_lbd * new_theta, dtype=torch.float).cuda()) == target):
                sign = -1

            # Untargeted case
            if (target is None and
                    self.model.predict_label(
                        x0 + torch.tensor(initial_lbd * new_theta, dtype=torch.float).cuda()) != y0):
                sign = -1

            queries += 1
            X[:, iii] = sign * u.reshape((dim,))

        Q = X.transpose().dot(X)
        q = -1 * np.ones((K,))
        G = np.diag(-1 * np.ones((K,)))
        h = np.zeros((K,))
        ### Use quad_qp solver 
        # alpha = solve_qp(Q, q, G, h)
        ### Use coordinate descent solver written by myself, avoid non-positive definite cases
        alpha = quad_solver(Q, q)
        sign_grad = (X.dot(alpha)).reshape(theta.shape)

        return sign_grad, queries

    def fine_grained_binary_search_local(self, model, x0, y0, theta, initial_lbd=1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd

        # still inside boundary
        if model.predict_label(x0 + torch.tensor(lbd * theta, dtype=torch.float).cuda()) == y0:
            lbd_lo = lbd
            lbd_hi = lbd * 1.01
            nquery += 1
            while model.predict_label(x0 + torch.tensor(lbd_hi * theta, dtype=torch.float).cuda()) == y0:
                lbd_hi = lbd_hi * 1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd * 0.99
            nquery += 1
            while model.predict_label(x0 + torch.tensor(lbd_lo * theta, dtype=torch.float).cuda()) != y0:
                lbd_lo = lbd_lo * 0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if model.predict_label(x0 + torch.tensor(lbd_mid * theta, dtype=torch.float).cuda()) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def fine_grained_binary_search(self, model, x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best:
            if model.predict_label(x0 + torch.tensor(current_best * theta, dtype=torch.float).cuda()) == y0:
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
            if model.predict_label(x0 + torch.tensor(lbd_mid * theta, dtype=torch.float).cuda()) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def __call__(self, input_xi, label_or_target, targeted=False, distortion=None, seed=None,
                 svm=False, query_limit=4000, momentum=0.0, stopping=0.0001,
                 args=None):  # this line: dummy args to match signopt-lf
        if targeted:
            raise NotImplementedError
            # adv = self.attack_targeted(input_xi, label_or_target, target, distortion=distortion,
            #                            seed=seed, svm=svm, query_limit=query_limit, stopping=stopping)
        else:
            adv = self.attack_untargeted(input_xi, label_or_target, distortion=distortion,
                                         svm=svm, query_limit=query_limit, momentum=momentum,
                                         stopping=stopping)
        return adv
