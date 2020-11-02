from collections import OrderedDict
from itertools import count
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image
from art.config import ART_NUMPY_DTYPE
from torch.autograd import Variable
from tqdm import trange

from scaleadv.models.layers import Pool2d, RandomPool2d
from scaleadv.models.scaling import ScaleNet

EARLY_STOP_ITER = 200
EARLY_STOP_THRESHOLD = 0.999
TANH_TOLERANCE = 1 - 1e-6


class ScaleAttack(object):
    """This class implements Scaling attack with several variants.
    1. Common Attack
       Hide an arbitrary small image (possibly adversarial) into a large image.
    2. Adaptive Attack
       Like 1, but is robust to deterministic and non-deterministic defenses.

    Args:
        scale_net: scaling network of type `ScaleNet`.
        class_net: classification network of type `nn.Module`.
        pooling: pooling layer (defense) of type `Pool2d` (optional).

    Keyword Args:
        lr: step size for scaling attack.
        max_iter: maximum number of iterations for scaling attack.
        lam_inp: extra multiplier for L2 penalty of input space loss.
        nb_samples: number of samples to approximate EE(pooling).
        early_stop: stop optimization if loss has converged.
    """

    def __init__(
            self,
            scale_net: ScaleNet,
            class_net: nn.Module,
            pooling: Optional[Pool2d] = None,
            lr: float = 0.01,
            max_iter: int = 1000,
            lam_inp: float = 1.0,
            nb_samples: int = 1,
            early_stop: bool = True,
    ):
        if nb_samples < 1:
            raise ValueError(f'Expect at least one sample, but got {nb_samples}.')

        self.scale_net = scale_net
        self.class_net = class_net
        self.pooling = pooling
        self.lr = lr
        self.max_iter = max_iter
        self.lam_inp = lam_inp
        self.nb_samples = nb_samples
        self.early_stop = early_stop

    @staticmethod
    def img_to_tanh(x: torch.Tensor) -> torch.Tensor:
        x = (x * 2 - 1) * TANH_TOLERANCE
        x = torch.atanh(x)
        return x

    @staticmethod
    def tanh_to_img(x: torch.Tensor) -> torch.Tensor:
        x = (x.tanh() + 1) * 0.5
        return x

    def predict(self, x: torch.Tensor, scale: bool = False, pooling: bool = False, n: int = 1) -> np.ndarray:
        """Predict big/small image with pooling support.
        Args:
            x: input image of shape [1, 3, H, W].
            scale: True if input image needs to be scaled.
            pooling: True if you want to apply pooling before scaling.
            n: number of samples for the pooling layer.

        Returns:
            np.ndarray containing predicted labels (multiple for n > 1).
        """
        with torch.no_grad():
            if pooling:
                assert scale, 'Cannot apply pooling without scaling.'
                x = x.to(self.pooling.dev)  # to the device recommended by pooling
                x = self.pooling(x.repeat(n, 1, 1, 1)).cuda()
            if scale:
                x = self.scale_net(x)
            pred = self.class_net(x).argmax(1).cpu()

        return pred.numpy()

    def generate(
            self,
            src: np.ndarray,
            tgt: np.ndarray,
            adaptive: bool = False,
            mode: str = 'sample',
            test_freq: int = 0,
    ) -> np.ndarray:
        """Run scale-attack with given source and target images.

        Args:
            src: large source image, of shape [1, 3, H, W].
            tgt: small target image, of shape [1, 3, h, w].
            adaptive: True if run adaptive-attack against predefined pooling layer.
            mode: how to approximate the random pooling, only 'sample', 'average', and 'worst' supported now.
            test_freq: full test per `test` iterations, set 0 to disable it.

        Returns:
            np.ndarray: final large attack image

        Notes:
            1. 'worst' returns the worst result by up-sampling with linear interpolation.
               this solves both median and random defenses with "2\beta" kernel width.
            2. 'average' solves median defenses, but not sure for random defenses.
               do note that this returns worse results than solving median-filter directly.

        Todo:
            1. 'average' is now using hard-coded params.
        """
        # Check params
        for x in [src, tgt]:
            assert x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 3
            assert x.dtype == np.float32
        if adaptive is True:
            assert mode in ['sample', 'average', 'worst'], f'Unsupported adaptive mode "{mode}".'

        # Convert to tensor
        src = torch.as_tensor(src, dtype=torch.float32).cuda()
        tgt = torch.as_tensor(tgt, dtype=torch.float32).cuda()
        factor = np.sqrt(1. * src.numel() / tgt.numel())

        # Return worst case result
        if adaptive and mode == 'worst':
            x = nn.functional.interpolate(tgt, src.shape[2:], mode='bilinear')
            x = np.array(x.cpu(), dtype=ART_NUMPY_DTYPE)
            return x

        # Get predicted labels
        y_src = self.predict(src, scale=True).item()
        y_tgt = self.predict(tgt, scale=False).item()

        # Prepare attack vars
        var = Variable(torch.zeros_like(src), requires_grad=True)
        var.data = self.img_to_tanh(src.data)
        # best_init = nn.functional.interpolate(tgt, src.shape[-2:], mode='bilinear')
        # var.data = self.img_to_tanh(best_init.data)

        # Prepare attack optimizer
        optimizer = torch.optim.Adam([var], lr=self.lr)

        # Start attack
        desc = 'ScaleAttack' + (' (adaptive)' if adaptive else '')
        with trange(self.max_iter, desc=desc) as pbar:
            prev_loss = np.inf
            for i in pbar:
                # Get attack image (big)
                att = self.tanh_to_img(var)

                # Get defensed image (big)
                att_def = att
                if adaptive:
                    if mode == 'sample':
                        att_def = att_def.to(self.pooling.dev)
                        att_def = att_def.repeat(self.nb_samples, 1, 1, 1)
                        att_def = self.pooling(att_def).cuda()
                    elif mode == 'average':
                        p, k, s = 2, 5, 3
                        # p, k, s = 2, 5, 1
                        att_def = nn.functional.pad(att_def, [p, p, p, p], mode='reflect')
                        att_def = nn.functional.avg_pool2d(att_def, k, s)
                        # att_def = self.scale_net(att_def)

                # Get scaled image (small)
                if adaptive and mode == 'average':
                    inp = att_def
                else:
                    inp = self.scale_net(att_def)

                # Compute loss
                loss = OrderedDict()
                loss['BIG'] = (src - att).reshape(att.shape[0], -1).norm(2, dim=1).mean()
                loss['INP'] = (tgt - inp).reshape(inp.shape[0], -1).norm(2, dim=1).mean() * factor
                total_loss = loss['BIG'] + self.lam_inp * loss['INP']

                # Optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Logging
                loss['TOTAL'] = total_loss
                stats = OrderedDict({k: f'{v.cpu().item():.3f}' for k, v in loss.items()})
                with torch.no_grad():
                    pred = self.predict(inp)
                    if pred.shape[0] == 1:
                        stats['PRED'] = pred.item()
                    else:
                        for y in [y_src, y_tgt]:
                            stats[f'PRED-{y}'] = f'{np.mean(pred == y):.2%}'
                pbar.set_postfix(stats)

                # Test
                if test_freq and i % test_freq == 0:
                    pred = self.predict(att, scale=True, pooling=True)
                    for y in [y_src, y_tgt]:
                        print(f'Test {y}: {np.mean(pred == y):.2%}')
                    F.to_pil_image(att[0].cpu().detach()).save(f'ADV-{i:03d}.png')

                # Early stop
                if self.early_stop and i % EARLY_STOP_ITER == 0:
                    if total_loss > prev_loss * EARLY_STOP_THRESHOLD:
                        break
                    prev_loss = total_loss

        # Convert to numpy
        att = np.array(att.detach().cpu(), dtype=ART_NUMPY_DTYPE)
        return att

    def generate_L0(self, src: np.ndarray, tgt: np.ndarray):
        """Test only, did not pass the test yet.
        """
        # Check params
        for x in [src, tgt]:
            assert x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 3
            assert x.dtype == np.float32

        # Convert to tensors
        src = torch.as_tensor(src, dtype=torch.float32).cuda()
        tgt = torch.as_tensor(tgt, dtype=torch.float32).cuda()

        # Prepare attack params
        var = torch.autograd.Variable(torch.zeros_like(src), requires_grad=True)
        var.data = torch.atanh((src.data * 2 - 1) * (1 - self.tol))
        mask = torch.ones_like(src[0, 0], requires_grad=False)  # mask is consistent per channel
        optimizer = torch.optim.Adam([var], lr=self.lr)
        opt = None

        # Outer attack
        prev_loss_out = np.inf
        for i in range(100):
            # Inner attack
            print(f'Attack Epoch {i}, L0 = {mask.norm(0).cpu().item()}')
            with trange(self.max_iter, desc=f'ScaleAttack-{i}') as pbar:
                prev_loss = np.inf
                for j in pbar:
                    # forward & loss
                    att = src * (1 - mask) + (var.tanh() + 1) * 0.5 * mask
                    inp = self.scale_net(att)
                    loss = OrderedDict({
                        'BIG': (src - att).reshape(att.shape[0], -1).norm(2, dim=1).mean(),
                        'INP': (tgt - inp).reshape(inp.shape[0], -1).norm(2, dim=1).mean(),
                    })
                    total_loss = loss['BIG'] + self.lam_inp * loss['INP']

                    # optimize
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    # loggging
                    loss['TOTAL'] = total_loss
                    stats = OrderedDict({k: f'{v.cpu().item():.3f}' for k, v in loss.items()})
                    pbar.set_postfix(stats)

                    # early stop
                    if self.early_stop and j % EARLY_STOP_ITER == 0:
                        if total_loss > prev_loss * EARLY_STOP_THRESHOLD:
                            break
                        prev_loss = total_loss

            # early stop for outer attack
            if total_loss > prev_loss_out * 1.2:
                opt = att, inp
                break
            prev_loss_out = total_loss

            # apply L0 constraint
            for j in count():
                print(f'L0 ({j}) = {mask.norm(0).cpu().item()}')
                # forward with current mask
                att = src * (1 - mask) + (var.tanh() + 1) * 0.5 * mask
                inp = self.scale_net(att)
                loss_big = (src - att).reshape(att.shape[0], -1).norm(2, dim=1).mean()
                loss_inp = (tgt - inp).reshape(inp.shape[0], -1).norm(2, dim=1).mean()
                total_loss = loss_big + self.lam_inp * loss_inp
                if total_loss > prev_loss_out * 1.05:
                    break
                optimizer.zero_grad()
                total_loss.backward()
                # measure how much changed
                with torch.no_grad():
                    delta = torch.abs(var.grad * loss_big * mask).sum(dim=(0, 1))
                    nonzero_delta = delta[delta > 0].cpu()
                    if nonzero_delta.numel() == 0:
                        opt = att, inp
                        break
                    # update mask
                    tau = np.percentile(nonzero_delta, 10)
                    mask = torch.ones_like(mask)
                    mask[delta <= tau] = 0

            if opt is not None:
                break

        att, inp = opt
        att = np.array(att.detach().cpu(), dtype=ART_NUMPY_DTYPE)
        inp = np.array(inp.detach().cpu(), dtype=ART_NUMPY_DTYPE)
        return att, inp

    def generate_with_given_pooling(self,
                                    src: torch.Tensor,
                                    tgt: torch.Tensor,
                                    y_tgt: int,
                                    fix_pooling: torch.Tensor, ):
        # Convert to tensor
        # src = torch.as_tensor(src, dtype=torch.float32).cuda()
        # tgt = torch.as_tensor(tgt, dtype=torch.float32).cuda()
        # pool = torch.as_tensor(fix_pooling, dtype=torch.float32).cuda()

        # Prepare attack
        src_atanh = torch.atanh((src * 2 - 1) * (1 - self.tol))
        pool_atanh = torch.atanh((fix_pooling * 2 - 1) * (1 - self.tol))
        var = torch.autograd.Variable(torch.zeros_like(src), requires_grad=True)
        optimizer = torch.optim.Adam([var], lr=self.lr)
        y_tgt = torch.LongTensor([y_tgt]).repeat((self.nb_samples,)).cuda()

        # Start attack
        with trange(self.max_iter, desc='ScaleAttack') as pbar:
            for _ in pbar:
                # Get attack image (big)
                att = torch.tanh(src_atanh + var) * 0.5 + 0.5

                # Get perturbed defensed image (big)
                att_def = torch.tanh(pool_atanh + var) * 0.5 + 0.5

                # Get scaled image (small)
                inp = self.scale_net(att_def)

                # Get prediction
                with torch.no_grad():
                    pred = self.class_net(inp)

                # Compute loss
                loss = OrderedDict()
                loss['BIG'] = (src - att).reshape(att.shape[0], -1).norm(2, dim=1).mean()
                loss['INP'] = (tgt - inp).reshape(inp.shape[0], -1).norm(2, dim=1).mean()
                # loss['CLS'] = nn.functional.cross_entropy(pred, y_tgt, reduction='mean')
                # total_loss = loss['BIG'] + self.lam_inp * loss['INP'] + loss['CLS']
                total_loss = loss['BIG'] + self.lam_inp * loss['INP']  # + loss['CLS']

                # Optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Logging
                loss['TOTAL'] = total_loss
                stats = OrderedDict({k: f'{v.cpu().item():.3f}' for k, v in loss.items()})
                with torch.no_grad():
                    if pred.shape[0] == 1:
                        stats['PRED'] = pred.argmax(1)[0].cpu().item()
                    else:
                        acc = (pred.argmax(1) == 100).float().mean().cpu().item()
                        stats['PRED-100'] = f'{acc:.2%}'
                        acc = (pred.argmax(1) == 200).float().mean().cpu().item()
                        stats['PRED-200'] = f'{acc:.2%}'
                pbar.set_postfix(stats)

        # Convert to numpy
        # att = np.array(att.detach().cpu(), dtype=ART_NUMPY_DTYPE)
        # inp = np.array(inp.detach().cpu(), dtype=ART_NUMPY_DTYPE)
        return att.detach(), inp.detach()

    def generate_test(self,
                      src: np.ndarray,
                      tgt: np.ndarray,
                      y_tgt: int):

        att = torch.as_tensor(src, dtype=torch.float32).cuda()
        tgt = torch.as_tensor(tgt, dtype=torch.float32).cuda()
        for i in range(10):
            # att = torch.as_tensor(att, dtype=torch.float32).cpu()
            with torch.no_grad():
                att_batch = self.pooling(att.cpu().repeat(self.nb_samples, 1, 1, 1)).cuda()
            # test
            import torchvision.transforms.functional as F
            for j, po in enumerate(att_batch[:10]):
                F.to_pil_image(po.cpu().detach()).save(f'test-{i}-{j}.png')
            att, inp = self.generate_with_given_pooling(att, tgt, y_tgt, att_batch)
            # test
            pred = self.test(att, n=self.nb_samples)
            print(f'Test 100: {np.mean(pred == 100):.2%}')
            print(f'Test 200: {np.mean(pred == 200):.2%}')
            Image.fromarray(np.round(att.cpu().detach().numpy()[0] * 255).astype(np.uint8).transpose((1, 2, 0))).save(
                f'test-{i}.png')
        return att, inp

    def generate_with_given_pooling_shadow(self,
                                           src: np.ndarray,
                                           tgt: np.ndarray,
                                           y_tgt: int,
                                           fix_pooling: np.ndarray, ):
        # Convert to tensor
        src = torch.as_tensor(src, dtype=torch.float32).cuda()
        tgt = torch.as_tensor(tgt, dtype=torch.float32).cuda()
        pool = torch.as_tensor(fix_pooling, dtype=torch.float32).cuda()

        # Prepare attack
        var = torch.autograd.Variable(torch.rand_like(src) - 0.5, requires_grad=True)
        optimizer = torch.optim.Adam([var], lr=self.lr)
        y_tgt = torch.LongTensor([y_tgt]).repeat((self.nb_samples,)).cuda()

        # Start attack
        with trange(self.max_iter, desc='ScaleAttack') as pbar:
            for _ in pbar:
                # Get perturbed defensed image (big)
                att = pool + var

                # Get scaled image (small)
                inp = self.scale_net(att)

                # Get prediction
                pred = self.class_net(inp)

                # Compute loss
                loss = OrderedDict()
                # loss['BIG'] = (src - att).reshape(att.shape[0], -1).norm(2, dim=1).mean()
                loss['INP'] = (tgt - inp).reshape(inp.shape[0], -1).norm(2, dim=1).mean()
                loss['CLS'] = nn.functional.cross_entropy(pred, y_tgt, reduction='mean')
                total_loss = loss['BIG'] + self.lam_inp * loss['INP'] + loss['CLS']

                # Optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Logging
                loss['TOTAL'] = total_loss
                stats = OrderedDict({k: f'{v.cpu().item():.3f}' for k, v in loss.items()})
                with torch.no_grad():
                    if pred.shape[0] == 1:
                        stats['PRED'] = pred.argmax(1)[0].cpu().item()
                    else:
                        acc = (pred.argmax(1) == 100).float().mean().cpu().item()
                        stats['PRED-100'] = f'{acc:.2%}'
                        acc = (pred.argmax(1) == 200).float().mean().cpu().item()
                        stats['PRED-200'] = f'{acc:.2%}'
                pbar.set_postfix(stats)

        # Convert to numpy
        att = np.array(att.detach().cpu(), dtype=ART_NUMPY_DTYPE)
        inp = np.array(inp.detach().cpu(), dtype=ART_NUMPY_DTYPE)
        return att, inp
