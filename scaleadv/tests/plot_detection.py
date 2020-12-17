import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

from scaleadv.defenses.detection import Unscaling, MinimumFilter
from scaleadv.tests.gen_acc_vs_budget import get_dataset_by_ratio
from scaleadv.tests.scale_adv import *

# Params
RATIO = 3
NB_DATA = 100
DEFENSE = 'none'
MODE = None
EPS = 4 * RATIO
ITER = 100
EPS_STEP = 30. * EPS / ITER

# Load data
ds = get_dataset_by_ratio(ratio=RATIO, nb_data=NB_DATA)
ld = DataLoader(ds, batch_size=1, shuffle=False, num_workers=8)

# Load scaling
lib, algo = LIB_TYPE['cv'], ALGO_TYPE['linear']
src_size, inp_size = (224 * RATIO, 224 * RATIO, 3), (224, 224, 3)
scale_down = ScalingGenerator.create_scaling_approach(src_size, inp_size, lib, algo)
scale_up = ScalingGenerator.create_scaling_approach(inp_size, src_size, lib, algo)
mask = get_mask_from_cl_cr(scale_down.cl_matrix, scale_up.cl_matrix)

# Get pooling layer
kernel = src_size[0] // inp_size[0] * 2 - 1
pooling_args = kernel, 1, kernel // 2, mask
pooling = POOLING[DEFENSE](*pooling_args)
nb_samples = 1 if MODE is None else 20

# Load network
scale_net = ScaleNet(scale_down.cl_matrix, scale_down.cr_matrix).eval()
class_net = nn.Sequential(NormalizationLayer.from_preset('imagenet'), resnet50_imagenet('2')).eval()
scale_net = scale_net.cuda()
class_net = class_net.cuda()

# Get scale attack
scl_attack = ScaleAttack(scale_net, class_net, pooling)
attack_args = dict(norm=2, eps=EPS, eps_step=EPS_STEP, max_iter=ITER, targeted=False, batch_size=NUM_SAMPLES_PROXY,
                   verbose=False)


def plot(det):
    # Run attack
    score_src, score_att = [], []
    acc, rob = [], []
    with tqdm(ld) as pb:
        for i, (src, y) in enumerate(pb):
            src = src.cpu().numpy()
            y = y.item()
            att = scl_attack.generate(src, y, IndirectPGD, attack_args, y_tgt=None, mode=MODE, nb_samples=nb_samples,
                                      verbose=False)
            score_src.append(det.score(src))
            score_att.append(det.score(att))
            # test acc
            x = class_net(scale_net(torch.tensor(src).cuda())).argmax(1).cpu().item()
            acc.append(x == y)
            x = class_net(scale_net(pooling(torch.tensor(att).cuda()))).argmax(1).cpu().item()
            rob.append(x == y)
            pb.set_postfix({'acc': np.mean(acc), 'rob': np.mean(rob)})

    # Eval
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4), constrained_layout=True)
    for i, name in enumerate(['MSE', 'SSIM']):
        ss, sa = [list(zip(*arr))[i] for arr in [score_src, score_att]]
        sns.distplot(ss, kde=False, label='Benign', ax=axes[i])
        sns.distplot(sa, kde=False, label='Attack', ax=axes[i])
        axes[i].set_xlabel(name)
        if name == 'SSIM':
            axes[i].set_xlim(-0.05, 1.05)
        axes[i].legend()

    acc, rob = map(np.mean, [acc, rob])
    fig.suptitle(f'Compare {det.name.title()} Defense (acc = {acc:.2%}, rob = {rob:.2%})')
    fig.savefig(f'det-{det.name}.pdf')


# Get detection
det = [
    Unscaling(scale_down, scale_up, pooling),
    MinimumFilter(),
]
plot(det[1])
