from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import sys
import time

subscription_key = "975e816744ab406f93326ef3dce10e0d"
endpoint = "https://gao257-test.cognitiveservices.azure.com/"



cv_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
# image = open('test.1000.generate.none.att.none.inp.png', 'rb')
# image = open('test.1000.generate.none.att.none.big.png', 'rb')


def get(tag, name):
    image = open(name, 'rb')
    resp = cv_client.describe_image_in_stream(image)
    print(tag)
    if resp.captions:
        for caption in resp.captions:
            print("  '{}' with confidence {:.2f}%".format(caption.text, caption.confidence * 100))
    print()


if __name__ == '__main__':
    imgs = {
        # 'benign': '/Users/gy/Desktop/tt/test.102.generate.none.src.none.inp.png',
        # 'attack': '/Users/gy/Desktop/tt/test.102.generate.none.att.none.inp.png',
        # 'scale': '/Users/gy/Desktop/tt/test.102.generate.none.att.none.big.png',
        # '0': '/Users/gy/Desktop/tt/102.adv.eps_0.small.png',
        # '5': '/Users/gy/Desktop/tt/102.adv.eps_5.small.png',
        # '10': '/Users/gy/Desktop/tt/102.adv.eps_10.small.png',
    }

    for tag, name in imgs.items():
        get(tag, name)