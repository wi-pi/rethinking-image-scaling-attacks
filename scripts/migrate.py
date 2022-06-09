import os

import pandas as pd


def load(name):
    try:
        data = pd.read_pickle(name)
        df = pd.DataFrame(data, columns=['iter', 'Query', 'Perturbation']).set_index('iter')
        return df
    except FileNotFoundError:
        pass


def migrate(src, tgt, ratio):
    os.makedirs(os.path.dirname(tgt), exist_ok=True)
    df = load(src)
    if df is not None and not df.empty:
        df.Perturbation /= ratio
        df.to_csv(tgt, index=False)


def migrate_bb_imagenet_api_hsj():
    id_list = range(0, 100)

    for ratio in [1, 3]:
        for i in id_list:
            src = f'static/online_bb_api_{ratio}x/{i}.ratio_{ratio}.def_none.log'
            tgt = f'static/logs/imagenet_api_hsj_none_{ratio}x/{i}.csv'
            yield src, tgt, ratio


def migrate_bb_celeba_hsj():
    for i in range(0, 200, 2):
        src = f'static/celeba_bb_celeba_hsj_small/{i}.ratio_3.def_none.log'
        tgt = f'static/logs/celeba_celeba_hsj_none_1x/{i}.csv'
        yield src, tgt, 1

        for r in [3, 5]:
            src = f'static/celeba_bb_celeba_hsj_large/{i}.ratio_{r}.def_none.log'
            tgt = f'static/logs/celeba_celeba_hsj_none_{r}x/{i}.csv'
            yield src, tgt, r


def migrate_bb_celeba_hsj_ablation():
    for i in range(0, 200, 5):
        for x, y in zip(['bad', 'good'], ['eq1', 'eq2']):
            src = f'static/celeba_bb_celeba_hsj_3x_{x}_10k/{i}.ratio_3.def_none.log'
            tgt = f'static/logs/celeba_celeba_hsj_none_3x/{y}/{i}.csv'
            yield src, tgt, 3


def migrate_bb_imagenet_hsj():
    for i in range(0, 100):
        # ratio 1
        src = f'static/bb/{i}.ratio_1.def_none.log'
        tgt = f'static/logs/imagenet_imagenet_hsj_none_1x/{i}.csv'
        yield src, tgt, 1

        # ratio 3 none
        src = f'static/bb/{i}.ratio_3.def_none.log'
        tgt = f'static/logs/imagenet_imagenet_hsj_none_3x/{i}.csv'
        yield src, tgt, 3

        src = f'static/bb_badnoise/{i}.ratio_3.def_none.log'
        tgt = f'static/logs/imagenet_imagenet_hsj_none_3x/bad_noise/{i}.csv'
        yield src, tgt, 3

        # ratio 3 median
        src = f'static/bb/{i}.ratio_3.def_median.log'
        tgt = f'static/logs/imagenet_imagenet_hsj_median_3x/vanilla/{i}.csv'
        yield src, tgt, 3

        tag1 = 'badmedian', 'med19', 'med28', 'med3565', 'med37', 'med46'
        tag2 = 'bad_median', 'q19', 'q28', 'q3565', 'q37', 'q46'
        for t1, t2 in zip(tag1, tag2):
            src = f'static/bb_{t1}/{i}.ratio_3.def_median.log'
            tgt = f'static/logs/imagenet_imagenet_hsj_median_3x/{t2}/{i}.csv'
            yield src, tgt, 3


def migrate_bb_imagenet_opt():
    for i in range(0, 100):
        # ratio 1
        src = f'static/bb_opt_small/{i}.ratio_3.def_none.log'
        tgt = f'static/logs/imagenet_imagenet_opt_none_1x/{i}.csv'
        yield src, tgt, 1

        # ratio 3 none
        src = f'static/bb_opt_good/{i}.ratio_3.def_none.log'
        tgt = f'static/logs/imagenet_imagenet_opt_none_3x/{i}.csv'
        yield src, tgt, 3

        src = f'static/bb_opt_bad/{i}.ratio_3.def_none.log'
        tgt = f'static/logs/imagenet_imagenet_opt_none_3x/bad_noise/{i}.csv'
        yield src, tgt, 3

        # ratio 3 median
        src = f'static/bb_opt_good/{i}.ratio_3.def_median.log'
        tgt = f'static/logs/imagenet_imagenet_opt_median_3x/{i}.csv'
        yield src, tgt, 3

        src = f'static/bb_opt_bad/{i}.ratio_3.def_median.log'
        tgt = f'static/logs/imagenet_imagenet_opt_median_3x/bad_median/{i}.csv'
        yield src, tgt, 3


if __name__ == '__main__':
    for args in migrate_bb_imagenet_opt():
        migrate(*args)
