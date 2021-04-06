import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import os

from scaleadv.utils import set_ccs_font

BINS = [1000 * i for i in range(1, 26)]

def get_one(name):
    if os.path.exists(name):
        return pickle.load(open(name, 'rb'))
    raise NotImplementedError

def to_bins(data):
    results = [0] * len(BINS)
    j = 0
    for i, query, dist in data:
        if query >= BINS[j]:
            results[j] = dist
            j += 1
    assert j == len(BINS)
    return results


def get(i):
    try:
        vanilla = get_one(f'static/bb/{i}.ratio_1.def_none.log')
        scale_none = get_one(f'static/bb/{i}.ratio_3.def_none.log')
        scale_med = get_one(f'static/bb/{i}.ratio_3.def_median.log')
    except NotImplementedError:
        return None

    vanilla, scale_none, scale_med = map(to_bins, [vanilla, scale_none, scale_med])
    return vanilla, scale_none, scale_med


if __name__ == '__main__':
    data = []
    for i in range(101):
        res = get(i)
        if res is not None:
            data.append(res)

    print('data plotted:', len(data))
    vanilla, scale_none, scale_med = map(pd.DataFrame, zip(*data))

    x = list(range(1, 26))
    set_ccs_font(10)
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    BLUE, ORANGE, GREEN, RED = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]
    plt.figure(figsize=(3, 3), constrained_layout=True)
    plt.plot(x, vanilla.median(), marker='D', ms=4, lw=1.5, c='k', label='HopSkipJump')
    plt.plot(x, scale_none.median() / 3, marker='o', ms=4, lw=1.5, c=GREEN, label='Scale-Adv (None)')
    plt.plot(x, scale_med.median() / 3, marker='^', ms=4, lw=1.5, c=ORANGE, label='Scale-Adv (Median)')
    plt.legend(borderaxespad=0.5)
    plt.yscale('log')
    plt.xticks(list(range(0, 26, 5)), [f'{i}K' for i in range(0, 26, 5)])
    plt.xlabel('Number of Queries')
    plt.ylabel(r'$\ell_2$ Distortion')
    plt.grid(True)
    plt.savefig(f'l2-vs-queries.pdf')



