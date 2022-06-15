from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger


class DataKit(object):

    def __init__(self, query_budgets: np.ndarray, pert_budgets: np.ndarray | None):
        self.query_budgets = query_budgets
        self.pert_budgets = pert_budgets
        self.data = dict()

    def load(self, key: str, root: str | Path, file_list: list[str] | None = None):
        root = Path(root)
        logger.info(f'Loading data from "{root}".')
        if file_list is None:
            file_list = root.glob('*.csv')
        else:
            file_list = [root / file for file in file_list]

        collected, data = [], []
        for file in file_list:
            df = pd.read_csv(file)
            if df is None:
                continue

            query, pert = df.Query.values, df.Perturbation.values
            idx_budget = np.argmax(query >= self.query_budgets[..., None] * 1000, axis=1)
            data.append(pert[idx_budget])
            collected.append(file)

        data = np.stack(data, axis=1)  # dim = [query budgets, perturbation data at this query budget]
        self.data[key] = np.nan_to_num(data)
        return collected

    def pert_vs_query(self, key: str):
        return np.median(self.data[key], axis=1)

    def sar_vs_pert(self, key: str):
        return np.mean(self.data[key][..., None] <= self.pert_budgets, axis=1) * 100

    def load_consistent(self, keys_roots: list[tuple[str, str | Path]]):
        file_list_total = []

        # 1st round
        for key, root in keys_roots:
            file_list = self.load(key, root)
            file_list_total.append(file_list)

        # find common files
        shared_file_list = [{f.name for f in file_list} for file_list in file_list_total]
        shared_file_list = list(set.intersection(*shared_file_list))

        # 2nd round
        for key, root in keys_roots:
            self.load(key, root, file_list=shared_file_list)

    def load_multiple(self, keys_roots: list[tuple[str, str | Path]], consistent: bool = True):
        if consistent:
            self.load_consistent(keys_roots)
        else:
            for key, root in keys_roots:
                self.load(key, root)


class PlotKit(object):
    ROOT = Path('static/logs/')
    OUTPUT = Path('static/plots/')

    @staticmethod
    def set_style(font_size: int = 20):
        mpl.rcParams['font.family'] = "Times New Roman"
        mpl.rcParams['mathtext.fontset'] = "cm"
        mpl.rcParams['font.size'] = font_size
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.spines.top'] = False
        mpl.rcParams['axes.linewidth'] = 1.5
        for t in 'xy':
            mpl.rcParams[f'{t}tick.major.size'] = 5
            mpl.rcParams[f'{t}tick.major.width'] = 1.5
            mpl.rcParams[f'{t}tick.minor.size'] = 2.5
            mpl.rcParams[f'{t}tick.minor.width'] = 1.5

    @staticmethod
    def pert_vs_query(
        query_budgets: np.ndarray,
        x_ticks: Iterable[int],
        curve_configs: dict,
        save: str | None = None,
        consistent: bool = True,
    ):
        # Init figure
        plt.figure(figsize=(5, 5), constrained_layout=True)

        # Collect
        dk = DataKit(query_budgets=query_budgets, pert_budgets=None)
        keys_roots = [(k, PlotKit.ROOT / c.pop('dir')) for k, c in curve_configs.items()]
        dk.load_multiple(keys_roots, consistent=consistent)

        # Plot
        for key, config in curve_configs.items():
            plt.plot(query_budgets, dk.pert_vs_query(key), lw=3, label=key, **config)

        # Wrapup
        plt.xlabel('Queries Budget (#)')
        plt.xticks(list(x_ticks), [f'{i}K' for i in x_ticks])
        plt.xlim(0, max(x_ticks))
        plt.ylabel(r'Median Perturbation (scaled $\ell_2$)')
        plt.yscale('log')
        plt.legend(borderaxespad=0, fontsize=18)
        plt.grid(True, linewidth=1.5)

        # Save
        if save:
            plt.savefig(PlotKit.OUTPUT / save)
        else:
            plt.show()

    @staticmethod
    def sar_vs_pert(
        query_budgets: np.ndarray,
        pert_budgets: np.ndarray,
        x_ticks: np.ndarray,
        text_pos: list[int],
        curve_configs: dict,
        legend_loc: str,
        save: str | None = None,
        consistent: bool = True,
    ):
        # Init figure
        plt.figure(figsize=(5, 5), constrained_layout=True)
        text_kwargs = dict(fontsize=16, rotation_mode='anchor', bbox=dict(fc='white', ec='none', pad=0),
                           transform_rotates_text=True)

        # Collect
        dk = DataKit(query_budgets=query_budgets, pert_budgets=pert_budgets)
        keys_roots = [(k, PlotKit.ROOT / c.pop('dir')) for k, c in curve_configs.items()]
        dk.load_multiple(keys_roots, consistent=consistent)

        # Plot
        for key, config in curve_configs.items():
            sar_all = dk.sar_vs_pert(key)

            for q, sar, p, c in zip(query_budgets, sar_all, text_pos, ['k', 'b', 'r']):
                plt.plot(pert_budgets, sar, c=c, lw=3, label={'k': key}.get(c), **config)
                rot = np.degrees(np.arctan2(sar[p + 1] - sar[p], 0.4))
                plt.text(pert_budgets[p], sar[p], f'{q}K', c=c, rotation=rot, **text_kwargs)

        # Wrapup
        plt.xlabel(r'Perturbation Budget (scaled $\ell_2$)')
        plt.xticks(x_ticks)
        plt.xlim(0, max(x_ticks))
        plt.ylabel('Attack Success Rate (%)')
        plt.yticks(list(range(0, 101, 20)))
        plt.ylim(0, 100)
        plt.legend(borderaxespad=0, loc=legend_loc, fontsize=18)
        plt.grid(True, linewidth=1.5)

        # Save
        if save:
            plt.savefig(PlotKit.OUTPUT / save)
        else:
            plt.show()
