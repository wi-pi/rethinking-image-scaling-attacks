from pathlib import Path

import numpy as np
import pandas as pd


class DataKit(object):

    def __init__(self, query_budgets: np.ndarray, pert_budgets: np.ndarray):
        self.query_budgets = query_budgets
        self.pert_budgets = pert_budgets
        self.data = dict()

    def load(self, key: str, root: str | Path, file_list: list[str] | None = None):
        root = Path(root)
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

    def pert_wrt_query(self, key: str):
        return np.median(self.data[key], axis=1)

    def sar_wrt_pert(self, key: str):
        return np.mean(self.data[key][..., None] <= self.pert_budgets, axis=1) * 100

    def load_consistent(self, keys_roots: list[tuple[str, str | Path]]):
        file_list_total = []

        # 1st round
        for key, root in keys_roots:
            file_list = self.load(key, root)
            file_list_total.append(file_list)

        # find common files
        shared_file_list = [{f.name for f in file_list} for file_list in file_list_total]
        shared_file_list = set.intersection(*shared_file_list)

        # 2nd round
        for key, root in keys_roots:
            self.load(key, root, file_list=shared_file_list)


if __name__ == '__main__':
    # dk = DataKit(query_budgets=np.arange(1, 4), pert_budgets=np.arange(0, 2.1, 0.1))
    # dk.load('api_1x', 'static/logs/imagenet_api_hsj_none_1x')
    # dk.load('api_3x', 'static/logs/imagenet_api_hsj_none_3x')

    # dk = DataKit(query_budgets=np.arange(1, 21), pert_budgets=np.arange(0, 2.1, 0.1))
    # dk.load('face_1x', 'static/logs/celeba_celeba_hsj_none_1x')
    # dk.load('face_3x', 'static/logs/celeba_celeba_hsj_none_3x')
    # dk.load('face_5x', 'static/logs/celeba_celeba_hsj_none_5x')

    # dk = DataKit(query_budgets=np.arange(1, 11), pert_budgets=np.arange(0, 6.1, 0.5))
    # dk.load_consistent([
    #     ('face_3x_eq1', 'static/logs/celeba_celeba_hsj_none_3x/eq1'),
    #     ('face_3x_eq2', 'static/logs/celeba_celeba_hsj_none_3x/eq2')
    # ])


    import IPython as i; i.embed(using=False); exit()