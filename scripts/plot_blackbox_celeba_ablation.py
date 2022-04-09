import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def load_data(fmt):
    data = []
    for i in id_list:
        try:
            d = pickle.load(open(fmt.format(i), 'rb'))
        except Exception as err:
            continue

        if not d:
            continue

        _, query, pert = map(np.array, zip(*d))
        pos = np.argmax(query >= query_budgets[..., None] * 1000, axis=1)  # first occurrence
        data.append(pert[pos])

    return np.stack(data, axis=1)  # dim = [query, data of this query]


if __name__ == '__main__':
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    BLUE, ORANGE, GREEN, RED = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]

    id_list = range(0, 200, 10)
    query_budgets = np.arange(1, 4)

    data_hsj_3x_eq1 = load_data('static/celeba_bb_celeba_hsj_3x_bad/{}.ratio_3.def_none.log') / 3
    data_hsj_3x_eq2 = load_data('static/celeba_bb_celeba_hsj_large/{}.ratio_3.def_none.log') / 3
    data_hsj_3x_eq2 = data_hsj_3x_eq2[:, :len(data_hsj_3x_eq1)]

    print('data loaded:', data_hsj_3x_eq1.shape, data_hsj_3x_eq2.shape)

    for q in [1, 2, 3]:
        for d in [data_hsj_3x_eq1, data_hsj_3x_eq2]:
            pert = np.median(d[q - 1])
            sar = np.mean(d[q - 1] <= 2) * 100
            print(f'{pert:.2f} {sar:.1f}')
        print()
