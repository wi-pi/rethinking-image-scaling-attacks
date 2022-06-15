import imagesize
import numpy as np
import torch
import torchvision.transforms as T
from facenet_pytorch.models.mtcnn import MTCNN
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm

from src.datasets import get_imagenet, get_celeba
from src.datasets.transforms import Align, ToNumpy
from src.scaling import ScalingAPI


class DatasetHelper(Dataset):
    """
    Helper class to load dataset with a fixed scaling-ratio.
    """

    # Make sure all test images are downscaled from the same HR image (scale = 5).
    max_scale = 5

    def __init__(
        self,
        name: str,
        scale: int,
        base: int = 224,
        valid_samples_only: bool = True,
        min_scale: int | None = None,
    ):
        """
        Create a Dataset Helper instance.

        Args:
            name: Name of the dataset, imagenet or celeba.
            scale: Desired scaling ratio w.r.t the base size.
            base: Desired base size.
            valid_samples_only: Only use samples that are correctly classified.
            min_scale: Only use samples whose scaling ratio is at least this much.
        """
        # Basic
        self.valid_samples_only = valid_samples_only
        self.id_list = None
        if min_scale is not None:
            if name != 'imagenet':
                logger.warning(f'Ignored {min_scale=} for dataset {name}.')
            if valid_samples_only:
                raise ValueError('Cannot use valid_samples when min_scale is specified.')

        # Load data
        match name:
            case 'imagenet':
                # Filter images with min scale if specified
                if min_scale is not None:
                    def check(x):
                        return min(imagesize.get(x)) >= min_scale * base

                    logger.info(f'Pre-filtering dataset with minimum scaling ratio {min_scale}.')
                    dataset = get_imagenet(split='val', transform=None)
                    self.id_list = [i for i, (x, _) in enumerate(tqdm(dataset.imgs)) if check(x)]

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

        # Create naive id_list if not yet
        if self.id_list is None:
            self.id_list = list(range(len(self.dataset)))

        logger.info(f'Loaded dataset {name}: {scale = }, {base = }, size = {len(self)} ({min_scale=}).')

        # Load meta preprocessing (from max scale to desired scale)
        raw_size = base * self.max_scale
        out_size = base * scale
        self.meta_scaling = ScalingAPI((raw_size, raw_size), (out_size, out_size), 'cv', 'linear')

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, item):
        x, y = self.dataset[self.id_list[item]]
        x = self.meta_scaling(x)
        return x, y

    def sample(self, n: int, seed: int = 0):
        ids = np.random.default_rng(seed).permutation(len(self))[:n]
        logger.debug(f'Randomly sampled {len(ids)} (out of {len(self)}) images.')
        return ids
