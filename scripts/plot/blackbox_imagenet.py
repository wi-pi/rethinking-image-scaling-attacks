import os
from pathlib import Path

import numpy as np

from scripts.plot.utils import PlotKit

if __name__ == '__main__':
    PlotKit.ROOT = Path('static/logs/')
    PlotKit.OUTPUT = Path('static/plots/')
    os.makedirs(PlotKit.OUTPUT, exist_ok=True)
    PlotKit.set_style()

    hsj_name = 'hsj', 'HSJ', 'q28'
    opt_name = 'opt', 'Sign-OPT', ''
    for short, full, med in zip(*zip(hsj_name, opt_name)):
        # no defense
        PlotKit.pert_vs_query(
            query_budgets=np.arange(1, 26),
            x_ticks=range(0, 26, 5),
            curve_configs={
                f'{full} (HR)': dict(dir=f'imagenet_imagenet_{short}_none_3x', c='C2'),
                f'{full} (HR w/o SNS)': dict(dir=f'imagenet_imagenet_{short}_none_3x/bad_noise', c='C1'),
                f'{full} (LR)': dict(dir=f'imagenet_imagenet_{short}_none_1x', c='k', ls='--'),
            },
            save=f'pert-vs-query-{short}-none.pdf'
        )
        PlotKit.sar_vs_pert(
            query_budgets=np.array([10, 15, 20]),
            pert_budgets=np.arange(0, 2.1, 0.1),
            x_ticks=np.arange(0, 2.1, 0.5),
            text_pos=[10, 12, 18],
            curve_configs={
                f'{full} (HR)': dict(dir=f'imagenet_imagenet_{short}_none_3x', ls='-'),
                f'{full} (HR w/o SNS)': dict(dir=f'imagenet_imagenet_{short}_none_3x/bad_noise', ls='--'),
                f'{full} (LR)': dict(dir=f'imagenet_imagenet_{short}_none_1x', ls=':'),
            },
            legend_loc='lower right',
            save=f'sar-vs-pert-{short}-none.pdf'
        )

        # median defense
        PlotKit.pert_vs_query(
            query_budgets=np.arange(1, 26),
            x_ticks=range(0, 26, 5),
            curve_configs={
                f'{full} (HR)': dict(dir=f'imagenet_imagenet_{short}_median_3x/{med}', c='C2'),
                f'{full} (HR w/o improve)': dict(dir=f'imagenet_imagenet_{short}_median_3x/bad_median', c='C1'),
                f'{full} (LR)': dict(dir=f'imagenet_imagenet_{short}_none_1x', c='k', ls='--'),
            },
            save=f'pert-vs-query-{short}-median.pdf'
        )
        PlotKit.sar_vs_pert(
            query_budgets=np.array([10, 15, 20]),
            pert_budgets=np.arange(0, 2.1, 0.1),
            x_ticks=np.arange(0, 2.1, 0.5),
            text_pos=[10, 12, 18],
            curve_configs={
                f'{full} (HR)': dict(dir=f'imagenet_imagenet_{short}_median_3x/{med}', ls='-'),
                f'{full} (HR w/o improve)': dict(dir=f'imagenet_imagenet_{short}_median_3x/bad_median', ls='--'),
                f'{full} (LR)': dict(dir=f'imagenet_imagenet_{short}_none_1x', ls=':'),
            },
            legend_loc='upper left',
            save=f'sar-vs-pert-{short}-median.pdf'
        )

    # ablation of median quantile
    PlotKit.set_style(font_size=18)
    PlotKit.pert_vs_query(
        query_budgets=np.arange(1, 26),
        x_ticks=range(0, 26, 5),
        curve_configs={
                          fr'HSJ (HR, $a=0.{a}, b=0.{b}$)': dict(dir=f'imagenet_imagenet_hsj_median_3x/q{a}{b}')
                          for a, b in zip([4, 3, 2, 1], [6, 7, 8, 9])
                      } | {
                          'HSJ (HR w/o improve)': dict(dir='imagenet_imagenet_hsj_median_3x/bad_median', ls='--',
                                                       c='k'),
                          'HSJ (LR)': dict(dir='imagenet_imagenet_hsj_none_1x', ls=':', c='k')
                      },
        save='pert-vs-query-hsj-median-quantile.pdf'
    )
