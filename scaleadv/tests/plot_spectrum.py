import cv2
import matplotlib.pyplot as plt
import numpy as np

src = 'raw.png'
att = {
    'none': 'TEST.5000.generate.none.None.ATT.plain.big.png',
    'median': 'TEST.5000.generate.random.cheap.ATT.plain.big.png',
    'random': 'TEST.5000.generate.median.None.ATT.plain.big.png',
    'bad': './old_imgs/TEST.ScaleAttack.Common.att.png',
    'hide-med': 'TEST.5000.hide.median.None.ATT.plain.big.png',
}


def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def idealFilterLP(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            if distance((y, x), center) < D0:
                base[y, x] = 1
    return base


def plot(img, i):
    img_c1 = cv2.imread(img, 0)
    img_c2 = np.fft.fft2(img_c1)
    img_c3 = np.fft.fftshift(img_c2)
    plt.subplot(230 + 1 + i), plt.imshow(img_c1, "gray"), plt.axis('off')
    plt.subplot(230 + 2 + i), plt.imshow(np.log(1 + np.abs(img_c2)), "gray"), plt.axis('off')
    plt.subplot(230 + 3 + i), plt.imshow(np.log(1 + np.abs(img_c3)), "gray"), plt.axis('off')


for tag, name in att.items():
    plt.figure(figsize=(15, 10), constrained_layout=True)
    plot(src, 0)
    plot(name, 3)
    plt.savefig(f'det-{tag}.pdf')
