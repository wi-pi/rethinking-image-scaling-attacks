import os

import numpy as np

from scripts.plot.utils import PlotKit

if __name__ == '__main__':
    os.makedirs(PlotKit.OUTPUT, exist_ok=True)
    PlotKit.set_style()

    PlotKit.pert_vs_query(
        query_budgets=np.arange(0, 3.1, 0.2),
        x_ticks=range(0, 4, 1),
        curve_configs={
            'HSJ (HR)': dict(dir='imagenet_api_hsj_none_3x', c='C2', ls='-'),
            'HSJ (LR)': dict(dir='imagenet_api_hsj_none_1x', c='C1', ls='-'),
        },
        save='pert-vs-query-hsj-api.pdf',
    )
    
    PlotKit.sar_vs_pert(
        query_budgets=np.array([1, 2, 3]),
        pert_budgets=np.arange(0, 10.1, 0.5),
        x_ticks=np.arange(0, 10.1, 2.0),
        text_pos=[10, 12, 18],
        curve_configs={
            'HSJ (HR)': dict(dir='imagenet_api_hsj_none_3x', ls='-'),
            'HSJ (LR)': dict(dir='imagenet_api_hsj_none_1x', ls='--'),
        },
        legend_loc='lower right',
        save='sar-vs-pert-hsj-api.pdf',
    )
