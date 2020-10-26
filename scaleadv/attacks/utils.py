import numpy as np
import numpy.linalg as LA


def get_mask_from_cl_cr(cl: np.ndarray, cr: np.ndarray) -> np.ndarray:
    cli, cri = map(LA.pinv, [cl, cr])
    shape = (cli.shape[1], cri.shape[0])
    mask = cli @ np.ones(shape) @ cri
    return mask.round().astype(np.uint8)
