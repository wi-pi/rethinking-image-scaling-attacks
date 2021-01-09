import cv2
import matplotlib.pyplot as plt
import numpy as np

from scaleadv.utils import set_ccs_font

img = {
    'source': 'testcases/scaling-attack/source.png',
    'attack': 'testcases/scaling-attack/attack.png',
}

def plot(tag, i, j):
    name = img[tag]
    img_color = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
    img_c1 = cv2.imread(name, 0)
    img_c2 = np.fft.fft2(img_c1)
    img_c3 = np.fft.fftshift(img_c2)

    h, w = img_c2.shape
    plt.subplot(2, 2, i), plt.imshow(img_color, interpolation='none', extent=[0, h, 0, w])
    h, w = h // 2, w // 2
    plt.subplot(2, 2, j), plt.imshow(np.log(1 + np.abs(img_c3)), 'gray', interpolation='none', extent=[-h, h, -w, w])


if __name__ == '__main__':
    set_ccs_font(14)
    plt.figure(figsize=(6, 6), tight_layout=True)
    plot('source', 1, 2)
    plot('attack', 3, 4)
    plt.savefig('test.pdf')

