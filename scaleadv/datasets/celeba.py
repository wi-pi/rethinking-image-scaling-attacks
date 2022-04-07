from functools import partial
from pathlib import Path

import torch
from torchvision.datasets import CelebA

DATASET_PATH = Path('static/datasets/celeba')

FULL_ATTRS = ('5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
              'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
              'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
              'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
              'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
              'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young')
SUBSET_ATTRS = ('Attractive', 'Bags_Under_Eyes', 'Bangs', 'Chubby', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open',
                'Mustache', 'Smiling', 'Wearing_Lipstick', 'Young')


def get_celeba(root: Path = DATASET_PATH, split='test', attrs=SUBSET_ATTRS, transform=None):
    # get attr index
    index = torch.tensor(list(map(FULL_ATTRS.index, attrs)))
    target_transform = partial(torch.take, index=index)

    # get dataset
    root = Path(root)
    CelebA.base_folder = root.name
    dataset = CelebA(root=root.parent, split=split, transform=transform, target_transform=target_transform)
    return dataset
