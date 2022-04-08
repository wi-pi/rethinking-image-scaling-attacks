import io
import os
from base64 import b64encode
from typing import List, Optional

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from tencentcloud.common import credential
from tencentcloud.tiia.v20190529 import tiia_client, models
from tencentcloud.tiia.v20190529.models import DetectLabelItem


class OnlineModel(object):
    ID, KEY = map(os.environ.get, ['TENCENT_ID', 'TENCENT_KEY'])

    def __init__(self):
        cred = credential.Credential(self.ID, self.KEY)
        self.client = tiia_client.TiiaClient(cred, region='ap-shanghai')

    def predict(self, x: torch.Tensor, y_true: str) -> np.ndarray:
        # Query
        prediction = self.query(x)

        # Get confidence of true label
        pos_score = 0
        for pred in prediction:
            if pred.Name == y_true:
                pos_score = pred.Confidence
                break

        # Get confidence of a different label
        neg_score = np.inf
        for pred in prediction:
            if pred.Name != y_true:
                neg_score = pred.Confidence
                break

        # Construct a dummy logits
        output = np.array([[neg_score, pos_score]])
        return output

    def get_true_label(self, x: torch.Tensor) -> Optional[str]:
        label = self.query(x)[0]
        return label.Name if label.Confidence > 50 else None

    def query(self, x: torch.Tensor) -> List[DetectLabelItem]:
        # To PIL
        buf = io.BytesIO()
        F.to_pil_image(x).save(buf, format='png')

        # Send request
        req = models.DetectLabelRequest()
        req.ImageBase64 = b64encode(buf.getvalue()).decode()
        req.Scenes = ['CAMERA']

        # Get respond labels
        resp = self.client.DetectLabel(req)
        return resp.CameraLabels


if __name__ == '__main__':
    model = OnlineModel()

    file_list = [
        'testcases/scaling-attack/source.png',
        'testcases/scaling-attack/target.png',
        'testcases/scaling-attack/attack.png',
    ]

    for name in file_list:
        x = F.to_tensor(Image.open(name))
        y_true = model.get_true_label(x)
        y_pred = model.predict(x, y_true)
        print(y_true, y_pred)
