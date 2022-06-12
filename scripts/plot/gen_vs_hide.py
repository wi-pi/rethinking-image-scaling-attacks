import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from scripts.plot.utils import PlotKit


def load(query_budgets: Iterable[int]):
    pred, pert = [], []
    for q in query_budgets:
        pred.append(np.load(f'static/logs/hide-vs-gen/eval-pred-{q}.npy'))
        pert.append(np.load(f'static/logs/hide-vs-gen/eval-l2-{q}.npy'))

    pred, pert = map(np.stack, [pred, pert])
    return pred, pert


def plot(
    query_budgets: np.ndarray,
    pert_budgets: np.ndarray,
    pred_columns: list[int],
    pert_columns: list[int],
    x_ticks: np.ndarray,
    pos_gen: list[int],
    pos_hide: list[int],
    save: str | None = None,
):
    # Load data
    # pred: (query budgets, samples, (y_src, y_adv, y_none_hide, y_median_hide))   # we assume y_gen = y_adv
    # pert: (query budgets, samples, (l2_none_hide, l2_none_gen, l2_median_hide, l2_median_gen, l2_lr))
    pred, pert = load(query_budgets)

    # We only take the two columns we need (depending on the defense)
    # pred: dim = (query budgets, samples, three preds) -- (y_src, y_adv, y_att|y_ada)
    # pert: dim = (query budgets, samples, two perts) -- (l2_gen, l2_hide)
    pred = pred[..., pred_columns]
    pert = pert[..., pert_columns]

    # dim = (query, samples, gen & hide)
    pred_ok = np.not_equal(pred[..., 1:], pred[..., 0, None])
    # dim = (query budgets, samples, pert budgets, gen & hide)
    pert_ok = pert[..., None, :] <= pert_budgets[..., None]
    # dim = (query budgets, pert budgets, gen & hide)
    sar_all = np.mean(pred_ok[..., None, :] & pert_ok, axis=1) * 100

    # Init figure
    plt.figure(figsize=(5, 5), constrained_layout=True)
    text_kwargs = dict(fontsize=16, rotation_mode='anchor', bbox=dict(fc='white', ec='none', pad=0),
                       transform_rotates_text=True)

    def _pp(sar, query, c, ls, label, pos):
        plt.plot(pert_budgets, sar, ls=ls, lw=3, c=c, label=label)
        rot = np.degrees(np.arctan2(sar[pos + 1] - sar[pos], 0.4))
        plt.text(pert_budgets[pos], sar[pos], f'{query}K', c=c, rotation=rot, **text_kwargs)

    # Plot
    for q, sar, p1, p2, c in zip(query_budgets, sar_all, pos_gen, pos_hide, 'kbrg'):
        _pp(sar[:, 0], q, c, '-', {'k': 'Joint Attack'}.get(c), p1)
        _pp(sar[:, 1], q, c, '--', {'k': 'Sequential Attack'}.get(c), p2)

    # Wrapup
    plt.xlabel(r'Perturbation Budget (scaled $\ell_2$)')
    plt.xticks(x_ticks)
    plt.xlim(0, max(x_ticks))

    plt.ylabel('Attack Success Rate (%)')
    plt.yticks(list(range(0, 101, 20)))
    plt.ylim(0, 100)

    plt.legend(borderaxespad=0, loc='upper right', fontsize=18)
    plt.grid(True, linewidth=1.5)

    # Save
    if save:
        plt.savefig(PlotKit.OUTPUT / save)
    else:
        plt.show()


if __name__ == '__main__':
    os.makedirs(PlotKit.OUTPUT, exist_ok=True)
    PlotKit.set_style()

    plot(
        query_budgets=np.array([1, 5, 20]),
        pert_budgets=np.arange(0, 5.1, 0.1),
        pred_columns=[0, 1, 2],
        pert_columns=[1, 0],
        x_ticks=np.arange(0, 6, 1),
        pos_gen=[40, 20, 10],
        pos_hide=[40, 20, 10],
        save='gen-vs-hide-sar-vs-pert-none.pdf',
    )

    plot(
        query_budgets=np.array([1, 3, 5, 20]),
        pert_budgets=np.arange(0, 50.1, 1.0),
        pred_columns=[0, 1, 3],
        pert_columns=[3, 2],
        x_ticks=np.arange(0, 51, 10),
        pos_gen=[15, 6, 10, 10],
        pos_hide=[28, 30, 35, 40],
        save='gen-vs-hide-sar-vs-pert-median.pdf',
    )
