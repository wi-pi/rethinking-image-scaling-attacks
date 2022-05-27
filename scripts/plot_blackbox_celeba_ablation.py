import pickle

import numpy as np


def load_data(fmt):
    data = []
    collected_ids = []
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
        collected_ids.append(i)

    return collected_ids, np.stack(data, axis=1)  # dim = [query, data of this query]


if __name__ == '__main__':

    id_list = range(0, 200, 5)
    query_budgets = np.array([10])
    pert_budgets = np.arange(0, 6.1, 0.5)

    id_list_1, _ = load_data('static/celeba_bb_celeba_hsj_3x_bad_10k/{}.ratio_3.def_none.log')
    id_list_2, _ = load_data('static/celeba_bb_celeba_hsj_3x_good_10k/{}.ratio_3.def_none.log')
    id_list = list(set(id_list_1) & set(id_list_2))  # make sure we select the same ids
    _, data_hsj_3x_eq1 = load_data('static/celeba_bb_celeba_hsj_3x_bad_10k/{}.ratio_3.def_none.log')
    _, data_hsj_3x_eq2 = load_data('static/celeba_bb_celeba_hsj_3x_good_10k/{}.ratio_3.def_none.log')

    print('data loaded:', data_hsj_3x_eq1.shape, data_hsj_3x_eq2.shape)
    data_hsj_3x_eq1 /= 3
    data_hsj_3x_eq2 /= 3

    print()
    print('l2')
    for d in [data_hsj_3x_eq1, data_hsj_3x_eq2]:
        print(f'{np.median(d[-1]):.2f}')

    print()
    print('sar')
    print(' '.join(f'{x:5.1f}' for x in pert_budgets))
    for d in [data_hsj_3x_eq1, data_hsj_3x_eq2]:
        sar = np.mean(d[-1] <= pert_budgets[..., None], axis=1) * 100
        print(' '.join(f'{x:5.1f}' for x in sar))
