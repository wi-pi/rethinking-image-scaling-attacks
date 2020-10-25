"""
This module implements Scaling attack with invariants.
1. Common attack, with cross-entropy support.
2. Adaptive attack, against both deterministic and non-deterministic defenses.
"""
import numpy as np
import torch
from art.config import ART_NUMPY_DTYPE
from tqdm import trange

from scaleadv.datasets.imagenet import create_dataset
from scaleadv.models.scaling import ScaleNet
from scaleadv.tests.utils import resize_to_224x

EARLY_STOP_ITER = 200
EARLY_STOP_THRESHOLD = 0.999


class ScaleAttack(object):

    def __init__(self, scaling_network: ScaleNet, max_iter: int = 1000, lr: float = 0.01, early_stop: bool = True,
                 tol=1e-6):
        """
        Create a `ScaleAttack` instance.

        Args:
            max_iter: Maximum number of iterations.
            lr: Learning rate.
            early_stop: Stop optimization if loss has converged.
            tol: Tolerance when converting to tanh space.
        """
        self.scaling = scaling_network
        self.max_iter = max_iter
        self.lr = lr
        self.early_stop = early_stop
        self.tol = tol

    def generate(self, src: np.ndarray, tgt: np.ndarray):
        # Convert to tensor
        src = torch.as_tensor(src, dtype=torch.float32).cuda()
        tgt = torch.as_tensor(tgt, dtype=torch.float32).cuda()

        # Prepare attack
        var = torch.autograd.Variable(src.clone().detach(), requires_grad=True)
        var.data = torch.atanh((var.data * 2 - 1) * (1 - self.tol))
        optimizer = torch.optim.Adam([var], lr=self.lr)

        # Start attack
        with trange(self.max_iter, desc='ScaleAttack') as pbar:
            prev_loss = np.inf
            for i in pbar:
                att = (var.tanh() + 1) * 0.5
                inp = self.scaling(att)
                # Compute loss
                loss_big = torch.norm(src - att, p=2)
                loss_inp = torch.norm(tgt - inp, p=2)
                loss = loss_big + 2.0 * loss_inp
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Logging
                pbar.set_postfix({
                    'TOTAL': f'{loss.cpu().item():.3f}',
                    'BIG': f'{loss_big.cpu().item():.3f}',
                    'INP': f'{loss_inp.cpu().item():.3f}',
                })
                # Early stop
                if self.early_stop and i % EARLY_STOP_ITER == 0:
                    if loss > prev_loss * EARLY_STOP_THRESHOLD:
                        break
                    prev_loss = loss

        # Convert to numpy
        att = np.array(att.detach().cpu(), dtype=ART_NUMPY_DTYPE)
        inp = np.array(inp.detach().cpu(), dtype=ART_NUMPY_DTYPE)
        return att, inp


if __name__ == '__main__':
    import torchvision.transforms as T
    from scaling.ScalingGenerator import ScalingGenerator
    from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
    from scaling.SuppScalingLibraries import SuppScalingLibraries

    # load data
    dataset = create_dataset(transform=None)
    src, _ = dataset[5000]
    tgt, _ = dataset[1000]
    src = resize_to_224x(src)
    src, tgt = map(np.array, [src, tgt])

    # load scaling
    lib = SuppScalingLibraries.CV
    algo = SuppScalingAlgorithms.LINEAR
    scaling = ScalingGenerator.create_scaling_approach(src.shape, (224, 224, 4), lib, algo)
    tgt = scaling.scale_image(tgt)

    # load attack
    src = np.array(src / 255, dtype=np.float32).transpose((2, 0, 1))[None, ...]
    tgt = np.array(tgt / 255, dtype=np.float32).transpose((2, 0, 1))[None, ...]
    scaling_net = ScaleNet(scaling.cl_matrix, scaling.cr_matrix).eval().cuda()
    attack = ScaleAttack(scaling_net)
    att, inp = attack.generate(src, tgt)

    # save figs
    f = T.Compose([lambda x: x[0], torch.tensor, T.ToPILImage()])
    for n in ['src', 'tgt', 'att', 'inp']:
        var = locals()[n]
        f(var).save(f'TEST.ScaleAttack.{n}.png')
