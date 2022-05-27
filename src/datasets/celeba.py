from pathlib import Path
from typing import Sequence

import torch
from torchvision.datasets import CelebA

DATASET_PATH = Path('static/datasets/celeba')

FULL_ATTRS = ('5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
              'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
              'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
              'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
              'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
              'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young')


def get_celeba(split='test', attrs: Sequence[str] = FULL_ATTRS, transform=None):
    # get subset attr index
    attr_index = list(map(FULL_ATTRS.index, attrs))

    # define target transform
    def target_transform(y: torch.Tensor):
        y = torch.take(y, index=torch.tensor(attr_index)).squeeze(dim=-1)
        if y.ndim == 0:
            y = y.item()  # return scalar if not multi-class

        return y

    # get dataset
    root = Path(DATASET_PATH)
    CelebA.base_folder = root.name
    dataset = CelebA(root=str(root.parent), split=split, transform=transform, target_transform=target_transform)

    return dataset
