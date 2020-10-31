"""
This module implements Scaling attack with variants.
1. Common attack, with cross-entropy support.
2. Adaptive attack, against both deterministic and non-deterministic defenses.
"""
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from art.config import ART_NUMPY_DTYPE
from tqdm import trange
import torchvision.transforms.functional as F
from scaleadv.models.layers import Pool2d, RandomPool2d
from scaleadv.models.scaling import ScaleNet

EARLY_STOP_ITER = 200
EARLY_STOP_THRESHOLD = 0.999


class ScaleAttack(object):

    def __init__(
            self,
            scale_net: ScaleNet,
            pooling: Optional[Pool2d] = None,
            class_net: Optional[nn.Module] = None,
            max_iter: int = 1000,
            lr: float = 0.01,
            lam_inp: float = 1.0,
            nb_samples: int = 1,
            early_stop: bool = True,
            tol=1e-6):
        """
        Create a `ScaleAttack` instance.

        Args:
            scale_net: scaling network
            pooling: pooling layer as a defense
            class_net: classification network
            max_iter: maximum number of iterations.
            lr: learning rate
            lam_inp: lambda for loss_inp
            nb_samples: number of samples if pooling is not deterministic
            early_stop: stop optimization if loss has converged
            tol: tolerance when converting to tanh space
        """
        if pooling is not None and nb_samples < 1:
            raise ValueError(f'Expect at least one sample of the pooling layer, but got {nb_samples}.')

        self.scale_net = scale_net
        self.pooling = pooling
        self.class_net = class_net
        self.max_iter = max_iter
        self.lr = lr
        self.lam_inp = lam_inp
        self.nb_samples = nb_samples
        self.early_stop = early_stop
        self.tol = tol

    def generate(self,
                 src: np.ndarray,
                 tgt: np.ndarray,
                 use_pooling: bool = False,
                 use_ce: bool = False,
                 y_tgt: int = None):
        # Check params
        for x in [src, tgt]:
            assert x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 3
            assert x.dtype == np.float32

        # Convert to tensor
        src = torch.as_tensor(src, dtype=torch.float32).cuda()
        tgt = torch.as_tensor(tgt, dtype=torch.float32).cuda()

        # Prepare attack
        var = torch.autograd.Variable(src.clone().detach(), requires_grad=True)
        var.data = torch.atanh((var.data * 2 - 1) * (1 - self.tol))
        optimizer = torch.optim.Adam([var], lr=self.lr)
        if use_ce:
            y_tgt = torch.LongTensor([y_tgt]).repeat((self.nb_samples,)).cuda()

        # Start attack
        with trange(self.max_iter, desc='ScaleAttack') as pbar:
            prev_loss = np.inf
            for i in pbar:
                # Get attack image (big)
                att = (var.tanh() + 1) * 0.5

                # Get defensed image (big)
                att_def = att
                if use_pooling:
                    if isinstance(self.pooling, RandomPool2d):
                        att_def = att_def.cpu()
                    att_def = att_def.repeat(self.nb_samples, 1, 1, 1)
                    att_def = self.pooling(att_def).cuda()

                # Get scaled image (small)
                inp = self.scale_net(att_def)

                # Compute loss
                loss = OrderedDict()
                loss['BIG'] = (src - att).reshape(att.shape[0], -1).norm(2, dim=1).mean()
                loss['INP'] = (tgt - inp).reshape(inp.shape[0], -1).norm(2, dim=1).mean()
                if use_ce:
                    pred = self.class_net(inp)
                    loss['CLS'] = nn.functional.cross_entropy(pred, y_tgt, reduction='mean')
                total_loss = loss['BIG'] + self.lam_inp * loss['INP'] + loss.get('CLS', 0)

                # Optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Logging
                loss['TOTAL'] = total_loss
                stats = OrderedDict({k: f'{v.cpu().item():.3f}' for k, v in loss.items()})
                if use_ce:
                    with torch.no_grad():
                        if pred.shape[0] == 1:
                            stats['PRED'] = pred.argmax(1)[0].cpu().item()
                        else:
                            acc = (pred.argmax(1) == 100).float().mean().cpu().item()
                            stats['PRED-100'] = f'{acc:.2%}'
                            acc = (pred.argmax(1) == 200).float().mean().cpu().item()
                            stats['PRED-200'] = f'{acc:.2%}'
                pbar.set_postfix(stats)

                # Early stop
                if self.early_stop and i % EARLY_STOP_ITER == 0:
                    if total_loss > prev_loss * EARLY_STOP_THRESHOLD:
                        break
                    prev_loss = total_loss

                # Test
                if i % 10 == 0:
                    pred = self.test(att, n=self.nb_samples)
                    print(f'Test 100: {np.mean(pred == 100):.2%}')
                    print(f'Test 200: {np.mean(pred == 200):.2%}')
                    F.to_pil_image(att[0].cpu().detach()).save(f'ADV-{i:03d}.png')

        # Convert to numpy
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
                #total_loss = loss['BIG'] + self.lam_inp * loss['INP'] + loss['CLS']
                total_loss = loss['BIG'] + self.lam_inp * loss['INP']# + loss['CLS']

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
            Image.fromarray(np.round(att.cpu().detach().numpy()[0] * 255).astype(np.uint8).transpose((1,2,0))).save(f'test-{i}.png')
        return att, inp


    def test(self, x: np.ndarray, n: int):
        x = torch.as_tensor(x, dtype=torch.float32)
        with torch.no_grad():
            xs = self.pooling(x.cpu().repeat(n, 1, 1, 1)).cuda()
            xs = self.scale_net(xs)
            pred = self.class_net(xs).argmax(1)
        return pred.cpu().numpy()


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
