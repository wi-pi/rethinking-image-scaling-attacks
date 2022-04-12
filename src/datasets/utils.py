import numpy as np
import torch
import torchvision.transforms as T
from facenet_pytorch.models.mtcnn import MTCNN
from loguru import logger
from torch.utils.data import Dataset

from src.datasets import get_imagenet, get_celeba
from src.datasets.transforms import Align, ToNumpy
from src.scaling import ScalingAPI


class DatasetHelper(Dataset):
    # Make sure all test images are downscaled from the same HR image.
    max_scale = 5

    def __init__(self, name: str, scale: int, base: int = 224, valid_samples_only: bool = True):
        self.valid_samples_only = valid_samples_only

        # Load data
        match name:
            case 'imagenet':
                transform = T.Compose([Align(base, self.max_scale), T.ToTensor(), ToNumpy(batch=False)])
                self.dataset = get_imagenet(split='val', transform=transform)
                if valid_samples_only:
                    self.id_list = np.load('static/meta/valid_ids.imagenet.nature.npy')

            case 'celeba':
                _mtcnn = MTCNN(image_size=base * self.max_scale, post_process=False)

                def mtcnn(x):
                    if (x := _mtcnn(x)) is None:
                        return torch.zeros(3, self.max_scale * base, self.max_scale * base)
                    return x

                transform = T.Compose([mtcnn, ToNumpy(batch=False), lambda x: x / 255])
                self.dataset = get_celeba(attrs=['Mouth_Slightly_Open'], transform=transform)
                if valid_samples_only:
                    self.id_list = np.load('static/meta/valid_ids.celeba.Mouth_Slightly_Open.npy')

            case _:
                raise NotImplementedError(f'Unknown dataset "{name}".')

        logger.info(f'Loaded dataset {name}: {scale = }, {base = }, size = {len(self)}.')

        # Load meta preprocessing (from max scale to desired scale)
        raw_size = base * self.max_scale
        out_size = base * scale
        self.meta_scaling = ScalingAPI((raw_size, raw_size), (out_size, out_size), 'cv', 'linear')

    def __len__(self):
        return len(self.id_list) if self.valid_samples_only else len(self.dataset)

    def __getitem__(self, item):
        valid_id = self.id_list[item] if self.valid_samples_only else item
        x, y = self.dataset[valid_id]
        x = self.meta_scaling(x)
        return x, y
