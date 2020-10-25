from math import ceil

from PIL import Image


def resize_to_224x(img: Image.Image, more: int = 1):
    w, h = img.size
    w = 224 * ceil(w / 224) * more
    h = 224 * ceil(h / 224) * more
    return img.resize((w, h))
