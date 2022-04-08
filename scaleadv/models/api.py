import io
import os
from base64 import b64encode
from typing import List, Optional

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from loguru import logger
from tencentcloud.common import credential
from tencentcloud.tiia.v20190529 import tiia_client, models
from tencentcloud.tiia.v20190529.models import DetectLabelItem


class OnlineModel(object):
    ID, KEY = map(os.environ.get, ['TENCENT_ID', 'TENCENT_KEY'])

    def __init__(self):
        cred = credential.Credential(self.ID, self.KEY)
        self.client = tiia_client.TiiaClient(cred, region='ap-shanghai')
        self.current_label = None
        self.nb_query = 0

    def set_current_sample(self, x: np.ndarray) -> Optional[str]:
        true_label = self.get_true_label(x)
        self.current_label = true_label
        self.nb_query = 0
        logger.debug(f'Get true label {true_label}')
        return true_label

    def predict(self, x: np.ndarray) -> np.ndarray:
        # Query
        prediction = self.query(x)
        self.nb_query += 1

        # Get confidence of true label
        pos_score = 0
        for pred in prediction:
            if pred.Name == self.current_label:
                pos_score = pred.Confidence
                break

        # Get confidence of a different label
        neg_score = np.inf
        for pred in prediction:
            if pred.Name != self.current_label:
                neg_score = pred.Confidence
                break

        # Construct a dummy logits
        output = np.array([[neg_score, pos_score]])
        return output

    def get_true_label(self, x: np.ndarray) -> Optional[str]:
        label = self.query(x)[0]
        return label.Name if label.Confidence > 20 else None

    def query(self, x: np.ndarray) -> List[DetectLabelItem]:
        if x.ndim == 4:
            if x.shape[0] != 1:
                raise ValueError(f'Does not support batch size > 1, got {x.shape}.')
            x = x[0]

        # To PIL
        buf = io.BytesIO()
        F.to_pil_image(torch.from_numpy(x)).save(buf, format='png')

        # Send request
        req = models.DetectLabelRequest()
        req.ImageBase64 = b64encode(buf.getvalue()).decode()
        req.Scenes = ['CAMERA']

        # Get respond labels
        prediction = self.client.DetectLabel(req).CameraLabels
        msg = ', '.join(f'{pred.Name} {pred.Confidence}' for i, pred in enumerate(prediction) if i < 5)
        logger.debug(msg)
        return prediction


if __name__ == '__main__':
    model = OnlineModel()

    file_list = [
        'testcases/scaling-attack/source.png',
        'testcases/scaling-attack/target.png',
        'testcases/scaling-attack/attack.png',
    ]

    for name in file_list:
        x = F.to_tensor(Image.open(name)).numpy()
        y_true = model.set_current_sample(x)
        y_pred = model.predict(x)
        print(y_true, y_pred)
