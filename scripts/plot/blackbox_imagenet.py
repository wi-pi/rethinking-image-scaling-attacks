import os

import numpy as np

from scripts.plot.utils import PlotKit

if __name__ == '__main__':
    os.makedirs(PlotKit.OUTPUT, exist_ok=True)
    PlotKit.set_style()

    names = [
        ('hsj', 'HSJ'),
        ('opt', 'Sign-OPT'),
    ]
    defenses = [
        ('none', 'SNS', 'bad_noise'),
        ('median', 'improve', 'bad_median'),
    ]
    for short, full in names:
        for defense, tag, ablation in defenses:
            PlotKit.pert_vs_query(
                query_budgets=np.arange(1, 26),
                x_ticks=range(0, 26, 5),
                curve_configs={
                    f'{full} (HR)': dict(dir=f'imagenet_imagenet_{short}_{defense}_3x', c='C2'),
                    f'{full} (HR w/o {tag})': dict(dir=f'imagenet_imagenet_{short}_{defense}_3x/{ablation}', c='C1'),
                    f'{full} (LR)': dict(dir=f'imagenet_imagenet_{short}_none_1x', c='k', ls='--'),
                },
                save=f'pert-vs-query-{short}-{defense}.pdf'
            )
            PlotKit.sar_vs_pert(
                query_budgets=np.array([10, 15, 20]),
                pert_budgets=np.arange(0, 2.1, 0.1),
                x_ticks=np.arange(0, 2.1, 0.5),
                text_pos=[10, 12, 18],
                curve_configs={
                    f'{full} (HR)': dict(dir=f'imagenet_imagenet_{short}_{defense}_3x', ls='-'),
                    f'{full} (HR w/o {tag})': dict(dir=f'imagenet_imagenet_{short}_{defense}_3x/{ablation}', ls='--'),
                    f'{full} (LR)': dict(dir=f'imagenet_imagenet_{short}_none_1x', ls=':'),
                },
                legend_loc='lower right',
                save=f'sar-vs-pert-{short}-{defense}.pdf'
            )

    # ablation of median quantile
    base = {
        'HSJ (HR w/o improve)': dict(dir='imagenet_imagenet_hsj_median_3x/bad_median', ls='--', c='k'),
        'HSJ (LR)': dict(dir='imagenet_imagenet_hsj_none_1x', ls=':', c='k')
    }
    advanced = {
        fr'HSJ (HR, $a=0.{a}, b=0.{b}$)': dict(dir=f'imagenet_imagenet_hsj_median_3x/q{a}{b}')
        for a, b in zip([4, 3, 2, 1], [6, 7, 8, 9])
    }
    PlotKit.set_style(font_size=18)
    PlotKit.pert_vs_query(
        query_budgets=np.arange(1, 26),
        x_ticks=range(0, 26, 5),
        curve_configs=advanced | base,
        save='pert-vs-query-hsj-median-quantile.pdf'
    )
