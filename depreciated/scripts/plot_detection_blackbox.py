import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import trange

from depreciated.scaleadv import get_imagenet
from depreciated.scaleadv import Align
from depreciated.scaleadv import POOLING_MAPS
from depreciated.scaleadv import Unscaling
from depreciated.scaleadv import ScalingLayer
from depreciated.scaleadv import ScalingAPI
from depreciated.scaleadv.utils import set_ccs_font

"""
python -m scripts.plot_detection_blackbox [none|median] [ratio] [max queries]
"""
# Params
DEFENSE, RATIO, QUERIES = sys.argv[1:]
RATIO, QUERIES = map(int, [RATIO, QUERIES])

# Load data
transform = T.Compose([Align(224, RATIO), T.ToTensor(), lambda x: np.array(x)[None, ...]])
dataset = get_imagenet(f'val_{RATIO}', transform)
id_list = pickle.load(open(f'static/meta/valid_ids.model_none.scale_{RATIO}.pkl', 'rb'))[::4]

# Load scaling
src_size, inp_size = (224 * RATIO, 224 * RATIO), (224, 224)
scale_down = ScalingAPI(src_size, inp_size, 'cv', 'linear')
scale_up = ScalingAPI(inp_size, src_size, 'cv', 'linear')

# Load networks
scaling_layer = ScalingLayer.from_api(scale_down).cuda()
pooling_layer = POOLING_MAPS[DEFENSE].auto(RATIO * 2 - 1, scale_down.mask).cuda()


def plot(det):
    # Run attack
    score_src, score_att = [], []
    bb = 'bb_med37' if DEFENSE == 'median' else 'bb'
    with trange(100) as pb:
        for i in pb:
            # find number of iterations
            try:
                data = pickle.load(open(f'static/{bb}/{i}.ratio_{RATIO}.def_{DEFENSE}.log', 'rb'))
            except FileNotFoundError:
                continue
            for it, query, _ in data:
                if query >= QUERIES:
                    break

            # load images
            src, _ = dataset[id_list[i]]
            att = F.to_tensor(Image.open(f'static/{bb}/{i}.ratio_{RATIO}.def_{DEFENSE}.{it:02d}.png')).numpy()

            # compute scores
            score_src.append(det.score(src))
            score_att.append(det.score(att[None]))
    print(len(score_src))

    # Eval
    fig, axes = plt.subplots(ncols=2, figsize=(4, 2), constrained_layout=True)
    for i, name in enumerate(['MSE', 'SSIM']):
        ss, sa = [list(zip(*arr))[i] for arr in [score_src, score_att]]
        sns.distplot(ss, kde=False, label='Benign', ax=axes[i])
        sns.distplot(sa, kde=False, label='Attack', ax=axes[i])
        axes[i].set_xlabel(name)
        if name == 'SSIM':
            axes[i].set_xlim(-0.05, 1.05)
        axes[i].legend(frameon=False, borderaxespad=0, loc=i + 1)

    fig.suptitle(f'{det.name.title()} Defense ({QUERIES // 1000}K Queries)')
    fig.savefig(f'det-bb-{DEFENSE}-{det.name}.{QUERIES}.{RATIO}.pdf')

    from IPython import embed; embed(using=False); exit()


# Get detection
det = [
    Unscaling(scale_down, scale_up, pooling_layer),
    # MinimumFilter(),
]
set_ccs_font(12)
for d in det:
    plot(d)
