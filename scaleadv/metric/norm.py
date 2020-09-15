import torch

def _norm(x, y, p):
    return torch.norm(x - y, p)


def Linf(x, y):
    return _norm(x, y, p=float('inf'))


def L2(x, y):
    return _norm(x, y, p=2)

