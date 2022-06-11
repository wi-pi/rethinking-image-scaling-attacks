import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torchvision.transforms as T
from tqdm import tqdm

from scaleadv.datasets import get_imagenet
from scaleadv.datasets.transforms import Align
from scaleadv.defenses import POOLING_MAPS
from scaleadv.defenses.detection import Unscaling, MinimumFilter
from scaleadv.evaluate.utils import ImageManager, DataManager
from scaleadv.models import ScalingLayer
from scaleadv.scaling import ScalingAPI
from scaleadv.utils import set_ccs_font, get_id_list_by_ratio

"""
python -m scripts.plot_detection [generate|hide] [none|median|uniform] [ratio] [eps]
"""
# Params
ATTACK, DEFENSE, RATIO, EPS = sys.argv[1:]
RATIO, EPS = map(int, [RATIO, EPS])
ITER = 100
EPS_STEP = 30. * EPS / ITER
FIELD = {'none': 'Y', 'median': 'Y_MED', 'uniform': 'Y_RND'}[DEFENSE]

# Load data
transform = T.Compose([Align(224, RATIO), T.ToTensor(), lambda x: np.array(x)[None, ...]])
dataset = get_imagenet(f'val_{RATIO}', transform)
id_list = pickle.load(open(f'static/meta/valid_ids.model_2.scale_{RATIO}.pkl', 'rb'))
id_list = get_id_list_by_ratio(id_list, RATIO)

# Load scaling
src_size, inp_size = (224 * RATIO, 224 * RATIO), (224, 224)
scale_down = ScalingAPI(src_size, inp_size, 'cv', 'linear')
scale_up = ScalingAPI(inp_size, src_size, 'cv', 'linear')
im = ImageManager(scale_down)
dm = DataManager(scale_down)

# Load networks
scaling_layer = ScalingLayer.from_api(scale_down).cuda()
pooling_layer = POOLING_MAPS[DEFENSE].auto(RATIO * 2 - 1, scale_down.mask).cuda()


def plot(det):
    # Run attack
    score_src, score_att = [], []
    acc, rob = [], []
    with tqdm(id_list) as pb:
        for i in pb:
            stat = dm.load_att(i, EPS, DEFENSE, ATTACK)
            if stat is None or stat['src']['Y'] != dataset.targets[i]:
                continue
            src, y = dataset[i]
            att = im.load_att(i, EPS, DEFENSE, ATTACK)
            score_src.append(det.score(src))
            score_att.append(det.score(att))
            # test acc
            acc.append(y == stat['src']['Y'])
            rob.append(y == stat['att'][FIELD])
            pb.set_postfix({'acc': np.mean(acc), 'rob': np.mean(rob)})
        print(len(acc))

    # Eval
    fig, axes = plt.subplots(ncols=2, figsize=(6, 3), constrained_layout=True)
    for i, name in enumerate(['MSE', 'SSIM']):
        ss, sa = [list(zip(*arr))[i] for arr in [score_src, score_att]]
        sns.distplot(ss, kde=False, label='Benign', ax=axes[i])
        sns.distplot(sa, kde=False, label='Attack', ax=axes[i])
        axes[i].set_xlabel(name)
        if name == 'SSIM':
            axes[i].set_xlim(-0.05, 1.05)
        axes[i].legend(frameon=False, borderaxespad=0)

    acc, rob = map(np.mean, [acc, rob])
    fig.suptitle(f'{det.name.title()} Defense (accuracy {acc:.2%}, robustness {rob:.2%})')
    fig.savefig(f'det-{ATTACK}-{DEFENSE}-{det.name}.{EPS}.{RATIO}.pdf')


# Get detection
det = [
    Unscaling(scale_down, scale_up, pooling_layer),
    MinimumFilter(),
]
set_ccs_font(14)
for d in det[:1]:
    plot(d)
