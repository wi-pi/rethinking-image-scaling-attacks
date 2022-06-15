import pickle
from base64 import b64encode
from concurrent.futures.process import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from tencentcloud.tiia.v20190529 import models
from tqdm import tqdm

from depreciated.scaleadv import get_imagenet
from depreciated.scaleadv.evaluate.utils import DataManager
from depreciated.scaleadv import ScalingAPI
from depreciated.scaleadv.utils import set_ccs_font, get_id_list_by_ratio

BLUE, ORANGE, GREEN, RED = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]


def get_one(filename):
    """Get prediction results from API.
    """
    req = models.DetectLabelRequest()
    req.ImageBase64 = b64encode(open(filename, 'rb').read()).decode()
    req.Scenes = ['CAMERA']
    resp = client.DetectLabel(req)
    return resp.CameraLabels


def get_many(filename_list, save=None, load=None):
    """Get prediction results for multiple files
    """
    if load is not None:
        data = pickle.load(open(load, 'rb'))
        return [data[f] for f in filename_list]

    data = {}
    with tqdm(desc=save, total=len(filename_list)) as pbar:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for filename, label in zip(filename_list, executor.map(get_one, filename_list)):
                data[filename] = label
                pbar.update(1)

    if save is not None:
        pickle.dump(data, open(save, 'wb'))
    return data


def get_all_data():
    all_data = {}

    # run benign
    filename_list = [dataset.imgs[i][0] for i in id_list]
    all_data['src'] = get_many(filename_list, load='./static/oakland_results/api-src.pkl')

    # run cw & scale
    for k in kappa_list:
        filename_list = [f'static/images/3.cv.linear/{i}.adv.eps_{k}.small.png' for i in id_list]
        all_data[f'cw-{k}'] = get_many(filename_list, load=f'./static/oakland_results/api-cw-{k}.pkl')
        filename_list = [f'static/images/3.cv.linear/{i}.generate.eps_{k}.pool_none.big.png' for i in id_list]
        all_data[f'scale-{k}'] = get_many(filename_list, load=f'./static/oakland_results/api-scale-{k}.pkl')

    return all_data


def get_all_score():
    data = get_all_data()

    def get_confidence(labels, true_label):
        for l in labels:
            if l.Name == true_label:
                return l.Confidence
        return 0

    # get true label and score
    y_true, score_true = zip(*[(labels[0].Name, labels[0].Confidence) for labels in data['src']])
    score_true = np.array(score_true, dtype=int)

    # get true label's score in cw
    score_cw = [[get_confidence(pred, y) for (pred, y) in zip(data[f'cw-{k}'], y_true)] for k in range(11)]
    score_cw = np.array(score_cw, dtype=int)

    # get true label's score in scale
    score_scale = [[get_confidence(pred, y) for (pred, y) in zip(data[f'scale-{k}'], y_true)] for k in range(11)]
    score_scale = np.array(score_scale, dtype=int)

    return score_true, score_cw, score_scale


if __name__ == '__main__':
    # ID, KEY = map(os.environ.get, ['TENCENT_ID', 'TENCENT_KEY'])
    # cred = credential.Credential(ID, KEY)
    # client = tiia_client.TiiaClient(cred, region='ap-shanghai')

    MAX_WORKERS = 16

    # setting
    dataset = get_imagenet(f'val_3', None)
    id_list = pickle.load(open(f'static/meta/valid_ids.model_2.scale_3.pkl', 'rb'))
    id_list = get_id_list_by_ratio(id_list, 3)[::2][:120]
    kappa_list = list(range(11))

    # get score
    st, sc, ss = get_all_score()

    """Get perturbation
    """
    # prepare
    src_shape = (224 * 3, 224 * 3)
    inp_shape = (224, 224)
    scaling_api = ScalingAPI(src_shape, inp_shape, 'cv', 'linear')
    dm = DataManager(scaling_api)
    dm2 = DataManager(scaling_api, tag='.cw_med_it100')
    get_adv_data = lambda e: [dm.load_adv(i, e) for i in id_list]
    get_att_data = lambda e, d: [dm.load_att(i, e, d, 'generate') for i in id_list]
    get_med_data = lambda e, d: [dm2.load_att(i, e, d, 'generate') for i in id_list]

    # get pert for cw
    pc = [[stat['adv']['L2'] for stat in get_adv_data(k)] for k in kappa_list]
    pc = np.array(pc, dtype=np.float32)

    # get pert for scale
    ps = [[stat['att']['L2'] for stat in get_att_data(k, 'none')] for k in kappa_list]
    ps = np.array(ps, dtype=np.float32) / 3

    # get pert for scale-median
    pm = [[stat['att']['L2'] for stat in get_med_data(k, 'median')] for k in kappa_list]
    pm = np.array(pm, dtype=np.float32) / 3

    # skip ss < 50
    ok = np.argwhere(st >= 50)[:, 0]
    sc, ss, pc, ps = map(lambda x: x[:, ok], [sc, ss, pc, ps])
    st = st[ok]

    """Plot score-vs-kappa
    """
    set_ccs_font(10)
    plt.figure(figsize=(3, 3), constrained_layout=True)
    plt.plot(kappa_list, np.median(ss, axis=1), marker='o', ms=4, lw=1.5, c=GREEN, label='C&W Attack (scaling)')
    plt.plot(kappa_list, np.median(sc, axis=1), marker='^', ms=4, lw=1.5, c=ORANGE, label='C&W Attack (vanilla)')
    plt.xlim(-0.5, 10.5)
    plt.xticks(kappa_list, fontsize=12)
    plt.xlabel(r'Confidence ($\kappa$)')
    plt.ylabel(r'Prediction Score (%)')
    plt.legend(borderaxespad=0.5)
    plt.grid(True)
    plt.savefig('api-score-vs-kappa.pdf')

    """Plot sar-vs-pert
    """
    budget = np.arange(0, 21, 0.5)
    sar_c = np.mean((pc[..., None] <= budget) * (sc[..., None] <= 10), axis=1) * 100
    sar_s = np.mean((ps[..., None] <= budget) * (ss[..., None] <= 10), axis=1) * 100
    plt.figure(figsize=(3, 3), constrained_layout=True)
    text_kwargs = dict(fontsize=8, rotation_mode='anchor', bbox=dict(fc='white', ec='none', pad=0),
                       transform_rotates_text=True)
    def _pp(sar, kappa, c, ls, label, pos):
        plt.plot(budget, sar[kappa], ls=ls, lw=1.5, c=c, label=label)
        dy = sar[kappa][pos + 3] - sar[kappa][pos]
        rot = np.degrees(np.arctan2(dy, 1))
        plt.text(budget[pos], sar[kappa][pos], rf'$\kappa={kappa}$', c=c, rotation=rot, **text_kwargs)

    # k=2
    _pp(sar_s, 3, 'k', '-', 'C&W Attack (scaling)', 15)
    _pp(sar_c, 3, 'k', ':', 'C&W Attack (vanilla)', 15)
    # k=5
    _pp(sar_s, 5, 'b', '-', None, 20)
    _pp(sar_c, 5, 'b', ':', None, 20)
    # k=10
    _pp(sar_s, 8, 'r', '-', None, 30)
    _pp(sar_c, 8, 'r', ':', None, 30)

    plt.xlim(-0.5, 20.5)
    plt.xticks(list(range(0, 21, 5)), fontsize=12)
    # plt.ylim(-2, 102)
    # plt.yticks(list(range(0, 101, 20)), fontsize=12)
    plt.xlabel(r'Perturbation Budget (scaled $\ell_2$)')
    plt.ylabel('Success Attack Rate (%)')
    plt.legend(borderaxespad=0.5, loc='upper left', fontsize=10)
    plt.grid(True)
    plt.savefig(f'api-sar-vs-pert.pdf')
