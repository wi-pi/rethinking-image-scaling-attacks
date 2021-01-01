import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from loguru import logger

from scaleadv.scaling import ScalingAPI


class ImageManager(object):
    root = Path('./static/images/')

    adv_fmt = '{i}.adv.eps_{eps}.{type}.png'.format
    att_fmt = '{i}.{attack}.eps_{eps}.pool_{defense}.{type}.png'.format

    def __init__(self, api: ScalingAPI):
        self.path = self.root / f'{round(api.ratio)}.{api.lib.name.lower()}.{api.alg.name.lower()}'
        os.makedirs(self.path, exist_ok=True)
        logger.info(f'Saving images to path "{self.path}".')

    def save_adv(self, i: int, eps: int, adv: np.ndarray, att: np.ndarray):
        name = self.adv_fmt(i=i, eps=eps, type='small')
        self.save(adv, name)
        name = self.adv_fmt(i=i, eps=eps, type='big')
        self.save(att, name)

    def load_adv(self, i: int, eps: int):
        name = self.adv_fmt(i=i, eps=eps, type='small')
        return self.load(name)

    def load_base(self, i: int, eps: int):
        name = self.adv_fmt(i=i, eps=eps, type='big')
        return self.load(name)

    def save_att(self, i: int, eps: int, defense: str, attack: str, att: np.ndarray):
        name = self.att_fmt(i=i, attack=attack, eps=eps, defense=defense, type='big')
        self.save(att, name)

    def load_att(self, i: int, eps: int, defense: str, attack: str):
        name = self.att_fmt(i=i, attack=attack, eps=eps, defense=defense, type='big')
        return self.load(name)

    def save(self, x: np.ndarray, filename: str):
        if x.ndim == 4:
            x = x.squeeze(0)
        x = torch.as_tensor(x, dtype=torch.float32)
        x = F.to_pil_image(x)
        x.save(str(self.path / filename))

    def load(self, filename: str):
        x = Image.open(str(self.path / filename))
        x = F.to_tensor(x)[None, ...]
        x = np.array(x)
        return x


class DataManager(object):
    root = Path('./static/results/')

    adv_fmt = '{i}.eps_{eps}.pkl'.format
    att_fmt = '{i}.{attack}.eps_{eps}.pool_{defense}.pkl'.format

    def __init__(self, api: ScalingAPI):
        self.path = self.root / f'{round(api.ratio)}.{api.lib.name.lower()}.{api.alg.name.lower()}'
        os.makedirs(self.path, exist_ok=True)
        logger.info(f'Saving results to path "{self.path}".')

    def save_adv(self, i: int, eps: int, data: Any):
        name = self.adv_fmt(i=i, eps=eps)
        self.save(data, name)

    def load_adv(self, i: int, eps: int):
        name = self.adv_fmt(i=i, eps=eps)
        return self.load(name)

    def save_att(self, i: int, eps: int, defense: str, attack: str, data: Any):
        name = self.att_fmt(i=i, attack=attack, eps=eps, defense=defense)
        self.save(data, name)

    def load_att(self, i: int, eps: int, defense: str, attack: str):
        name = self.att_fmt(i=i, attack=attack, eps=eps, defense=defense)
        return self.load(name)

    def save(self, data: Any, filename: str):
        pickle.dump(data, open(str(self.path / filename), 'wb'))

    def load(self, filename: str):
        pth = self.path / filename
        return pickle.load(open(str(pth), 'rb')) if pth.exists() else None
