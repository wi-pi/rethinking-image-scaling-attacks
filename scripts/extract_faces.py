import os
import pickle

import torch
import torchvision.transforms.functional as F
from facenet_pytorch import MTCNN
from torchvision.models import resnet34
from tqdm import trange

from scaleadv.datasets.celeba import get_celeba, SUBSET_ATTRS

if __name__ == '__main__':
    # data
    dataset = get_celeba(root='static/datasets/celeba', attrs=SUBSET_ATTRS, transform=None)

    # face extractor
    mtcnn = MTCNN(image_size=224 * 5)

    # model
    model = resnet34(num_classes=len(SUBSET_ATTRS)).cuda()
    ckpt = torch.load('static/models/celeba-res34.pth', map_location='cpu')
    model.load_state_dict(ckpt)

    # process
    y_true_all, y_pred_all = [], []
    for i in trange(len(dataset)):
        # face extraction
        x_raw, y = dataset[i]
        x_face = mtcnn(x_raw)
        if x_face is None:
            continue

        # downscale & test
        x_face_224 = F.resize(x_face, size=224).cuda()[None]
        y_pred = torch.where(torch.sigmoid(model(x_face_224)) > 0.5, 1, 0).cpu()

        # collect
        y_true_all.append(y)
        y_pred_all.append(y_pred[0])

    # analyze
    y_true_all = torch.stack(y_true_all)
    y_pred_all = torch.stack(y_pred_all)

    acc = y_true_all == y_pred_all
    os.makedirs('static/celeba_valid_ids', exist_ok=True)
    for i, attr in enumerate(SUBSET_ATTRS):
        print(f'{i:2d} {acc[:, i].float().mean().item():.2%} {attr}')
        id_list = acc[:, i].nonzero().T[0].tolist()
        pickle.dump(id_list, open(f'static/celeba_valid_ids/{attr}.pkl', 'wb'))
