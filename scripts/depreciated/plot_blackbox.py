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


def plot_pert_vs_queries(defense: str):
    set_ccs_font(10)
    plt.figure(figsize=(3, 3), constrained_layout=True)
    if defense == 'none':
        plt.plot(query_budgets, np.median(data_none, 1), ms=4, lw=1.5, c=GREEN, label=f'{attack_full} (HR)')
        plt.plot(query_budgets, np.median(data_badnoise, 1), ms=4, lw=1.5, c=ORANGE, label=f'{attack_full} (HR w/o SNS)')
    if defense == 'median':
        plt.plot(query_budgets, np.median(data_median, 1), ms=4, lw=1.5, c=GREEN, label=f'{attack_full} (HR)')
        plt.plot(query_budgets, np.median(data_badmedian, 1), ms=4, lw=1.5, c=ORANGE, label=f'{attack_full} (HR w/o efficiency)')

    plt.plot(query_budgets, np.median(data_hsj, 1), ms=4, ls='--', lw=1.5, c='k', label=f'{attack_full} (LR)')
    plt.legend(borderaxespad=0.5)
    plt.yscale('log')
    plt.xticks(list(range(0, 26, 5)), [f'{i}K' for i in range(0, 26, 5)])
    plt.xlabel('Number of Queries (#)')
    plt.ylabel(r'Perturbation (scaled $\ell_2$)')
    plt.grid(True)
    plt.savefig(f'bb-{attack_short}-{defense}-l2-vs-queries.pdf')


def plot_pert_vs_queries_median():
    set_ccs_font(10)
    plt.figure(figsize=(3, 3), constrained_layout=True)
    for a, b in [(1, 9), (2, 8), (3, 7), (4, 6)][::-1]:
        data_median = load_data('static/bb_med%d%d/{}.ratio_3.def_median.log' % (a, b)) / 3
        plt.plot(query_budgets, np.median(data_median, 1), label=rf'Smoothed Median ($a = 0.{a}, b = 0.{b}$)')
    plt.plot(query_budgets, np.median(data_hsj, 1), ls='--', c='k', label='HSJ Attack')
    plt.legend(borderaxespad=0.5, fontsize=9)
    plt.yscale('log')
    plt.xticks(list(range(0, 26, 5)), [f'{i}K' for i in range(0, 26, 5)])
    plt.xlabel('Number of Queries (#)')
    plt.ylabel(r'Perturbation (scaled $\ell_2$)')
    plt.grid(True)
    plt.savefig(f'bb-l2-vs-queries-median.pdf')


def plot_pert_vs_queries_sns():
    set_ccs_font(10)
    plt.figure(figsize=(3, 3), constrained_layout=True)
    plt.plot(query_budgets, np.median(data_none, 1), marker='o', ms=4, lw=1.5, c=GREEN, label='HSJ Attack (scaling w/ SNS)')
    plt.plot(query_budgets, np.median(data_badnoise, 1), marker='^', ms=4, lw=1.5, c=ORANGE, label='HSJ Attack (scaling w/o SNS)')
    plt.plot(query_budgets, np.median(data_hsj, 1), ls='--', ms=4, lw=1.5, c='k', label='HSJ Attack (vanilla)')
    plt.legend(borderaxespad=0.5)
    plt.yscale('log')
    plt.xticks(list(range(0, 26, 5)), [f'{i}K' for i in range(0, 26, 5)])
    plt.xlabel('Number of Queries (#)')
    plt.ylabel(r'Perturbation (scaled $\ell_2$)')
    plt.grid(True)
    plt.savefig(f'bb-opt-l2-vs-queries-sns.pdf')


def plot_pert_vs_queries_smooth():
    set_ccs_font(10)
    plt.figure(figsize=(3, 3), constrained_layout=True)
    plt.plot(query_budgets, np.median(data_median, 1), marker='o', ms=4, lw=1.5, c=GREEN, label='HSJ Attack (median w/ smooth)')
    plt.plot(query_budgets, np.median(data_badmedian, 1), marker='^', ms=4, lw=1.5, c=ORANGE, label='HSJ Attack (median w/o smooth)')
    plt.plot(query_budgets, np.median(data_hsj, 1), ls='--', ms=4, lw=1.5, c='k', label='HSJ Attack (vanilla)')
    plt.legend(borderaxespad=0.5)
    plt.yscale('log')
    plt.xticks(list(range(0, 26, 5)), [f'{i}K' for i in range(0, 26, 5)])
    plt.xlabel('Number of Queries (#)')
    plt.ylabel(r'Perturbation (scaled $\ell_2$)')
    plt.grid(True)
    plt.savefig(f'bb-opt-l2-vs-queries-smooth.pdf')


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

    # plot query 5K
    q, c, p = 10, 'k', 10
    if defense == 'none':
        _pp(data_none, q, c, '-', f'{attack_full} (HR)', p)
        _pp(data_badnoise, q, c, '--', f'{attack_full} (HR w/o SNS)', p)
    if defense == 'median':
        _pp(data_median, q, c, '-', f'{attack_full} (HR)', p)
        _pp(data_badmedian, q, c, '--', f'{attack_full} (HR w/o efficiency)', p)
    _pp(data_hsj, q, c, ':', f'{attack_full} (LR)', p)

    # plot query 10K
    q, c, p = 15, 'b', 12
    if defense == 'none':
        _pp(data_none, q, c, '-', None, p)
        _pp(data_badnoise, q, c, '--', None, p)
    if defense == 'median':
        _pp(data_median, q, c, '-', None, p)
        _pp(data_badmedian, q, c, '--', None, p)
    _pp(data_hsj, q, c, ':', None, p)

    # plot query 25K
    q, c, p = 20, 'r', 18
    if defense == 'none':
        _pp(data_none, q, c, '-', None, p)
        _pp(data_badnoise, q, c, '--', None, p)
    if defense == 'median':
        _pp(data_median, q, c, '-', None, p)
        _pp(data_badmedian, q, c, '--', None, p)
    _pp(data_hsj, q, c, ':', None, p)

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
    plt.savefig(f'bb-{attack_short}-{defense}-sar-vs-pert.pdf')


if __name__ == '__main__':
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    BLUE, ORANGE, GREEN, RED = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]

    id_list = range(0, 100)
    # query_budgets = np.array([0.1, 0.2, 0.5, 1, 5, 10])
    query_budgets = np.arange(1, 26)
    pert_budgets = np.arange(0, 2.1, 0.1)

    # load data (hsj)
    attack_full = 'HSJ'
    attack_short = 'hsj'
    data_hsj = load_data('static/bb/{}.ratio_1.def_none.log')
    data_none = load_data('static/bb/{}.ratio_3.def_none.log') / 3
    data_median = load_data('static/bb_med28/{}.ratio_3.def_median.log') / 3
    data_badnoise = load_data('static/bb_badnoise/{}.ratio_3.def_none.log') / 3
    data_badmedian = load_data('static/bb_badmedian/{}.ratio_3.def_median.log') / 3

    # load data (opt)
    # attack_full = 'Sign-OPT'
    # attack_short = 'opt'
    # data_hsj = load_data('static/bb_opt_small/{}.ratio_3.def_none.log')
    # data_none = load_data('static/bb_opt_good/{}.ratio_3.def_none.log') / 3
    # data_median = load_data('static/bb_opt_good/{}.ratio_3.def_median.log') / 3
    # data_badnoise = load_data('static/bb_opt_bad/{}.ratio_3.def_none.log') / 3
    # data_badmedian = load_data('static/bb_opt_bad/{}.ratio_3.def_median.log') / 3

    print('data loaded:', data_hsj.shape, data_none.shape, data_median.shape)

    # plot data
    for d in ['none', 'median']:
        plot_pert_vs_queries(d)
        plot_sar_vs_pert(d)
    # plot_pert_vs_queries_median()
    # plot_pert_vs_queries_sns()
    # plot_pert_vs_queries_smooth()
