from typing import Union

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image


def savefig(x: Union[np.ndarray, torch.Tensor], name: str):
    if isinstance(x, np.ndarray):
        # handle dims
        if x.ndim > 4:
            raise ValueError(f'Cannot save image of shape {x.shape}.')
        if x.ndim == 4:
            if x.shape[0] != 1:
                raise ValueError(f'Cannot save batch images of shape {x.shape}.')
            x = np.squeeze(x, axis=0)

        # handle channels
        if x.shape[0] != 3:
            if x.shape[2] != 3:
                raise ValueError(f'Cannot save image of shape {x.shape}.')
            # from channel-last to channel-first
            x = np.transpose(x, axes=(2, 0, 1))

        # handle value range
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.0
        elif np.issubdtype(x.dtype, np.floating):
            if np.max(x) > 1.0:
                x /= 255.0
        else:
            raise ValueError(f'Cannot save image of dtype {x.dtype}.')

        # save fig
        to_pil_image(torch.tensor(x)).save(name)
        return

    if isinstance(x, torch.Tensor):
        # handle dim
        if len(x.shape) > 4:
            raise ValueError(f'Cannot save image of shape {x.shape}.')
        if len(x.shape) == 4:
            if x.shape[0] != 1:
                raise ValueError(f'Cannot save image of shape {x.shape}.')
            x = torch.squeeze(x, dim=0)

        # handle channels
        if x.shape[0] != 3:
            if x.shape[2] != 3:
                raise ValueError(f'Cannot save image of shape {x.shape}.')
            # from channel-last to channel-first
            x = torch.permute(x, dims=(2, 0, 1))

        # handle value range
        if torch.max(x) > 1.0:
            x /= 255.0

        # save fig
        to_pil_image(x).save(name)
        return

    raise NotImplementedError(f'Type {type(x)} not supported.')
