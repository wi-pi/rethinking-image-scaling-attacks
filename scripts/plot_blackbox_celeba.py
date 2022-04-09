import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from scaleadv.utils import set_ccs_font


def load_data(fmt):
    data = []
    for i in id_list:
        try:
            d = pickle.load(open(fmt.format(i), 'rb'))
        except:
            continue

        if not d:
            continue

        _, query, pert = map(np.array, zip(*d))
        pos = np.argmax(query >= query_budgets[..., None] * 1000, axis=1)  # first occurrence
        data.append(pert[pos])

    return np.stack(data, axis=1)  # dim = [query, data of this query]


def plot_sar_vs_pert(defense):
    set_ccs_font(10)
    plt.figure(figsize=(3, 3), constrained_layout=True)
    text_kwargs = dict(fontsize=8, rotation_mode='anchor', bbox=dict(fc='white', ec='none', pad=0),
                       transform_rotates_text=True)

    def _pp(data, query, c, ls, label, pos):
        sar = np.mean(data[query - 1] <= pert_budgets[..., None], axis=-1) * 100
        plt.plot(pert_budgets, sar, ls=ls, lw=1.5, c=c, label=label)
        rot = np.degrees(np.arctan2(sar[pos + 1] - sar[pos], 0.1))
        plt.text(pert_budgets[pos], sar[pos], f'{query}K', c=c, rotation=rot, **text_kwargs)

    # plot query 10K
    q, c, p = 10, 'k', 10
    if defense == 'none':
        _pp(data_hsj_1x, q, c, '-', f'HSJ (LR)', p)
        _pp(data_hsj_3x, q, c, '--', f'HSJ (HR, 3x)', p)
    _pp(data_hsj_5x, q, c, ':', f'HSJ (HR, 5x)', p)

    # plot query 15K
    q, c, p = 15, 'b', 12
    if defense == 'none':
        _pp(data_hsj_1x, q, c, '-', None, p)
        _pp(data_hsj_3x, q, c, '--', None, p)
    _pp(data_hsj_5x, q, c, ':', None, p)

    # plot query 20K
    q, c, p = 20, 'r', 18
    if defense == 'none':
        _pp(data_hsj_1x, q, c, '-', None, p)
        _pp(data_hsj_3x, q, c, '--', None, p)
    _pp(data_hsj_5x, q, c, ':', None, p)

    plt.xlim(-0.05, 2.05)
    plt.xticks(np.arange(0, 2.1, 0.5), fontsize=12)
    plt.ylim(-2, 102)
    plt.yticks(list(range(0, 101, 20)), fontsize=12)
    plt.xlabel(r'Perturbation Budget (scaled $\ell_2$)')
    plt.ylabel('Success Attack Rate (%)')
    if defense == 'none':
        plt.legend(borderaxespad=0.5, loc='lower right', fontsize=10)
    if defense == 'median':
        plt.legend(borderaxespad=0.5, loc='upper left', fontsize=10)
    plt.grid(True)
    plt.savefig(f'celeba-sar-vs-pert.pdf')


def plot_celeba():
    set_ccs_font(10)
    plt.figure(figsize=(3, 3), constrained_layout=True)

    plt.plot(query_budgets, np.median(data_hsj_1x, 1), ms=4, lw=1.5, c='k', label=fr'HSJ (LR, 224 $\times$ 224)')
    plt.plot(query_budgets, np.median(data_hsj_3x, 1), ms=4, lw=1.5, c=GREEN, label=fr'HSJ (HR, 672 $\times$ 672)')
    plt.plot(query_budgets, np.median(data_hsj_5x, 1), ms=4, lw=1.5, c=ORANGE, label=fr'HSJ (HR, 1120 $\times$ 1120)')

    plt.legend(borderaxespad=0.5)
    plt.yscale('log')
    plt.xticks(list(range(0, 21, 5)), [f'{i}K' for i in range(0, 21, 5)])
    plt.xlabel('Number of Queries (#)')
    plt.ylabel(r'Perturbation (scaled $\ell_2$)')
    plt.grid(True)
    plt.savefig(f'celeba-l2-vs-queries.pdf')


if __name__ == '__main__':
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    BLUE, ORANGE, GREEN, RED = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]

    id_list = range(0, 200, 2)
    query_budgets = np.arange(1, 21)
    pert_budgets = np.arange(0, 2.1, 0.1)

    data_hsj_1x = load_data('static/celeba_bb_celeba_hsj_small/{}.ratio_3.def_none.log')
    data_hsj_3x = load_data('static/celeba_bb_celeba_hsj_large/{}.ratio_3.def_none.log') / 3
    data_hsj_5x = load_data('static/celeba_bb_celeba_hsj_large/{}.ratio_5.def_none.log') / 5

    print('data loaded:', data_hsj_1x.shape, data_hsj_3x.shape, data_hsj_5x.shape)

    for q in [10, 15, 20]:
        for d in [data_hsj_1x, data_hsj_3x, data_hsj_5x]:
            pert = np.median(d[q - 1])
            sar = np.mean(d[q - 1] <= 2) * 100
            print(f'{pert:.2f} {sar:.1f}')
        print()

    # plot_celeba()
    # plot_sar_vs_pert('none')
