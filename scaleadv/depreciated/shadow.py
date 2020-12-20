from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from RandomizedSmoothing.utils.classification import get_acc
from RandomizedSmoothing.utils.log import j_header, j_print
from RandomizedSmoothing.utils.regularizers import get_tv, get_color, get_sim
from art.attacks.evasion import ShadowAttack
from art.config import ART_NUMPY_DTYPE
from art.utils import get_labels_np_array, check_and_transform_label_format
from torch.nn import DataParallel
from torchvision.models import resnet50
from tqdm import trange

from scaleadv.datasets.imagenet import create_dataset, IMAGENET_MEAN, IMAGENET_STD
from scaleadv.models.layers import NormalizationLayer


class SmoothAttack:
    def __init__(self, classifier: torch.nn.Module):
        self.classifier = classifier
        self._loss = torch.nn.CrossEntropyLoss(reduction='mean').cuda()

    def perturb(self, x: torch.Tensor, y: int, sigma: float = 0.5, batch: int = 400, steps: int = 300,
                duplicate_rgb: bool = False, lr: float = 0.1, tv_lam: float = 0.1, ch_lam: float = 20.0,
                dissim_lam: float = 10.0, print_stats: bool = False, **_) -> torch.Tensor:
        # torch.manual_seed(6247423)

        t = torch.rand_like(x[0] if duplicate_rgb else x).cuda() - 0.5
        t.requires_grad_()

        copy_size = (batch, 1, 1, 1)
        x_batch = x.repeat(copy_size).cuda()
        x_batch = x_batch + torch.randn_like(x_batch).cuda() * sigma
        y = torch.LongTensor([y]).cuda().repeat((batch,))

        if print_stats:
            j_header('step', 'acc', 'loss', 'cls', 'tv', 'col', 'dissim')
        for i in range(steps):
            ct = t.repeat((3, 1, 1)) if duplicate_rgb else t
            cur_in = x_batch + ct.repeat(copy_size)
            outputs = self.classifier(cur_in)
            acc, correct = get_acc(outputs, y)

            cl_loss = -torch.mean(self._loss(outputs, y))
            tv_loss = get_tv(ct)
            col_loss = get_color(ct)
            dissim_loss = 0 if duplicate_rgb else get_sim(ct)
            loss = cl_loss - tv_lam * tv_loss - ch_lam * col_loss - dissim_lam * dissim_loss
            loss.backward()

            t.data = t.data + lr * t.grad.data
            t.grad.data.zero_()

            if print_stats:
                j_print(i, acc, loss, cl_loss, tv_loss, col_loss, dissim_loss)

        ct = t.repeat((3, 1, 1)) if duplicate_rgb else t
        ct = ct.to(x.device)
        return torch.clamp((x + ct).data, 0.0, 1.0)

    def perturb_with_given_noise(self, x: torch.Tensor, noise: torch.Tensor, y: int, batch: int = 400, steps: int = 300,
                duplicate_rgb: bool = False, lr: float = 0.1, tv_lam: float = 0.1, ch_lam: float = 20.0,
                dissim_lam: float = 10.0, print_stats: bool = False, **_) -> Tuple[torch.Tensor, torch.Tensor]:

        t = torch.rand_like(x[0] if duplicate_rgb else x).cuda() - 0.5
        t.requires_grad_()

        copy_size = (batch, 1, 1, 1)
        x_batch = noise.cuda()
        y = torch.LongTensor([y]).cuda().repeat((batch,))

        if print_stats:
            j_header('step', 'acc', 'loss', 'cls', 'tv', 'col', 'dissim')
        for i in range(steps):
            ct = t.repeat((3, 1, 1)) if duplicate_rgb else t
            cur_in = x_batch + ct.repeat(copy_size)
            outputs = self.classifier(cur_in)
            acc, correct = get_acc(outputs, y)

            cl_loss = -torch.mean(self._loss(outputs, y))
            tv_loss = get_tv(ct)
            col_loss = get_color(ct)
            dissim_loss = 0 if duplicate_rgb else get_sim(ct)
            loss = cl_loss - tv_lam * tv_loss - ch_lam * col_loss - dissim_lam * dissim_loss
            loss.backward()

            t.data = t.data + lr * t.grad.data
            t.grad.data.zero_()

            if print_stats:
                j_print(i, acc, loss, cl_loss, tv_loss, col_loss, dissim_loss)

        ct = t.repeat((3, 1, 1)) if duplicate_rgb else t
        ct = ct.to(x.device)
        return torch.clamp((x + ct).data, 0.0, 1.0), ct.data

class CustomShadowAttack(ShadowAttack):

    def __init__(self, *args, **kwargs):
        super(CustomShadowAttack, self).__init__(*args, **kwargs)

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array. This requires a lot of memory, therefore it accepts
        only a single samples as input, e.g. a batch of size 1.

        :param x: An array of a single original input sample.
        :param y: An array of a single target label.
        :return: An array with the adversarial examples.
        """
        y = check_and_transform_label_format(y, self.estimator.nb_classes)

        if y is None:
            raise NotImplementedError
        else:
            self.targeted = True

        if x.shape[0] > 1 or y.shape[0] > 1:
            raise ValueError("This attack only accepts a single sample as input.")

        if x.ndim != 4:
            raise ValueError("Unrecognized input dimension. Shadow Attack can only be applied to image data.")

        x = x.astype(ART_NUMPY_DTYPE)
        x_batch = np.repeat(x, repeats=self.batch_size, axis=0).astype(ART_NUMPY_DTYPE)
        x_batch = x_batch + np.random.normal(scale=self.sigma, size=x_batch.shape).astype(ART_NUMPY_DTYPE)
        y_batch = np.repeat(y, repeats=self.batch_size, axis=0)

        perturbation = (
            np.random.uniform(
                low=self.estimator.clip_values[0], high=self.estimator.clip_values[1], size=x.shape
            ).astype(ART_NUMPY_DTYPE)
            - (self.estimator.clip_values[1] - self.estimator.clip_values[0]) / 2
        )

        # for _ in trange(self.nb_steps, desc="Shadow attack"):
        for i in range(self.nb_steps):
            print(f'{i}/{self.nb_steps}', end='\t')
            gradients_ce = np.mean(
                self.estimator.loss_gradient(x=x_batch + perturbation, y=y_batch, sampling=False)
                * (1 - 2 * int(self.targeted)),
                axis=0,
                keepdims=True,
            )
            gradients = gradients_ce - self._get_regularisation_loss_gradients(perturbation)
            perturbation += self.learning_rate * gradients

        x_p = x + perturbation
        x_adv = np.clip(x_p, a_min=self.estimator.clip_values[0], a_max=self.estimator.clip_values[1]).astype(
            ART_NUMPY_DTYPE
        )

        return x_adv

    def generate_with_given_noise(self, x: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """
        Args:
            x: An array of a single input image, shape is (1, C, H, W).
            noise: An array of a batch noise inputs, shape is (B, C, H, W).
        """
        # check format
        assert x.ndim == 4 and noise.ndim == 4
        assert x.shape[0] == 1 and noise.shape[0] >= 1
        assert x.shape[1] == 3 and noise.shape[1] == 3
        assert x.shape[2:] == noise.shape[2:]

        # prepare attack
        x = x.astype(ART_NUMPY_DTYPE)
        x_batch = noise.astype(ART_NUMPY_DTYPE)
        y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        #print('predict label is', y.argmax(1))
        y_batch = np.repeat(y, repeats=self.batch_size, axis=0)
        delta = (
                np.random.uniform(
                    low=self.estimator.clip_values[0], high=self.estimator.clip_values[1], size=x.shape
                ).astype(ART_NUMPY_DTYPE)
                - (self.estimator.clip_values[1] - self.estimator.clip_values[0]) / 2
        )

        # attack
        #for i in trange(self.nb_steps, desc="Shadow attack"):
        for i in range(self.nb_steps):
            print(f'{i}/{self.nb_steps}', end='\t')
            x_batch_adv = x_batch + delta
            loss = self.estimator.loss_gradient(x=x_batch_adv, y=y_batch, sampling=False) * (1 - 2 * int(self.targeted))
            gradients_ce = np.mean(loss, axis=0, keepdims=True)
            gradients = gradients_ce - self._get_regularisation_loss_gradients(delta)
            delta += self.learning_rate * gradients

        # finish attack
        x_p = x + delta
        x_adv = np.clip(x_p, a_min=self.estimator.clip_values[0], a_max=self.estimator.clip_values[1])
        x_adv = x_adv.astype(ART_NUMPY_DTYPE)
        return x_adv

    def _get_regularisation_loss_gradients(self, perturbation: np.ndarray) -> np.ndarray:
        import torch

        perturbation_t = torch.from_numpy(perturbation).to("cpu")
        perturbation_t.requires_grad = True

        x_t = perturbation_t[:, :, :, 1:] - perturbation_t[:, :, :, :-1]
        y_t = perturbation_t[:, :, 1:, :] - perturbation_t[:, :, :-1, :]

        loss_tv = (x_t * x_t).sum(dim=(1, 2, 3)) + (y_t * y_t).sum(dim=(1, 2, 3))

        if perturbation_t.shape[1] == 1:
            loss_s = 0.0
        elif perturbation_t.shape[1] == 3:
            loss_s = (
                    (perturbation_t[:, 0, :, :] - perturbation_t[:, 1, :, :]) ** 2
                    + (perturbation_t[:, 1, :, :] - perturbation_t[:, 2, :, :]) ** 2
                    + (perturbation_t[:, 0, :, :] - perturbation_t[:, 2, :, :]) ** 2
            ).norm(p=2, dim=(1, 2))

        loss_c = perturbation_t.abs().mean([2, 3]).norm(dim=1) ** 2
        #loss = torch.mean(self.lambda_tv * loss_tv + self.lambda_s * loss_s + self.lambda_c * loss_c)
        loss = self.lambda_tv * loss_tv + self.lambda_s * loss_s + self.lambda_c * loss_c
        print('\t'.join([f'{x.cpu().item():.3f}' for x in [loss_tv, loss_s, loss_c]]))
        loss.backward()
        gradients = perturbation_t.grad.numpy()
        return gradients




if __name__ == '__main__':
    # load data
    trans = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    dataset = create_dataset(transform=trans)
    _, x, y = dataset[5000]

    # load classifier
    model = nn.Sequential(NormalizationLayer(IMAGENET_MEAN, IMAGENET_STD), resnet50(pretrained=True)).eval()
    model = DataParallel(model).cuda()

    # test
    x = x.cuda()
    attack = SmoothAttack(model)
    x_adv = attack.perturb(x, y)

    for obj in [x, x_adv]:
        yp = model(obj[:, ...]).argmax(1)
        print(yp.cpu().item())

    from PIL import Image

    Image.fromarray(x_adv).save('test.png')
