import pickle

import numpy as np


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


if __name__ == '__main__':
    id_list = range(0, 100)
    query_budgets = np.arange(1, 4)
    pert_budgets = np.arange(0, 2.1, 0.1)

    data_api_1x = load_data('static/online_bb_api_1x/{}.ratio_1.def_none.log')
    data_api_1x = np.nan_to_num(data_api_1x)

    print('data loaded:', data_api_1x.shape)

    for q in [1, 2, 3]:
        for d in [data_api_1x]:
            pert = np.median(d[q - 1])
            sar = np.mean(d[q - 1] <= 2) * 100
            print(f'{pert:.2f} {sar:.1f}')
        print()
