import cv2
import matplotlib.pyplot as plt
import numpy as np

from scaleadv.utils import set_ccs_font

src = 'raw.png'
attack = 'generate'
if attack == 'hide':
    att = {
        'bad': './testcases/scaling-attack/attack.png',
        'none': './static/results/images/test.5000.hide.none.att.none.big.png',
        'median': './static/results/images/test.5000.hide.median.att.none.big.png',
        'random': './static/results/images/test.5000.hide.uniform.att.none.big.png',
    }
elif attack == 'generate':
    att = {
        'bad': './testcases/scaling-attack/attack.png',
        'none': './static/results/images/test.5000.generate.none.att.none.big.png',
        'median': './static/results/images/test.5000.generate.median.att.none.big.png',
        'random': './static/results/images/test.5000.generate.uniform.att.none.big.png',
    }
else:
    raise NotImplementedError


def plot(img, i, j):
    img_color = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    img_c1 = cv2.imread(img, 0)
    img_c2 = np.fft.fftshift(np.fft.fft2(img_c1))
    h, w = img_c2.shape
    plt.subplot(1, 4, i), plt.imshow(img_color, interpolation='none', extent=[0, h, 0, w])
    h, w = h // 2, w // 2
    plt.subplot(1, 4, j), plt.imshow(np.log(1 + np.abs(img_c2)), 'gray', interpolation='none', extent=[-h, h, -w, w])


def plot_all():
    def pp(tag, name, i):
        img_spatial = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
        img_spectrum = np.fft.fftshift(np.fft.fft2(cv2.imread(name, 0)))
        img_spectrum = np.log(1 + np.abs(img_spectrum))
        h, w = img_spectrum.shape
        plt.subplot(2, 5, i), plt.imshow(img_spatial, interpolation='none', extent=[0, h, 0, w])
        h, w = h // 2, w // 2
        ax = plt.subplot(2, 5, i + 5)
        plt.imshow(img_spectrum, 'gray', interpolation='none', extent=[-h, h, -w, w])
        ax.text(0.5, -0.1, tag, size=12, ha="center", transform=ax.transAxes)

    plt.figure(figsize=(25, 10), tight_layout=True)
    pp('Benign', src, 1)
    pp('Image-Scale Attack', att['bad'], 2)
    pp('Scale-Adv Attack (none)', att['none'], 3)
    pp('Scale-Adv Attack (median)', att['median'], 4)
    pp('Scale-Adv Attack (random)', att['random'], 5)
    plt.savefig(f'det-all.{attack}.pdf')


def plot_low_pass(tag):
    img_c1 = cv2.imread(att[tag], 0)
    img_c2 = np.fft.fftshift(np.fft.fft2(img_c1))
    h, w = [v // 2 for v in img_c1.shape]
    ts = [90, 95, 98, 99]
    plt.figure(figsize=(5 * len(ts), 5), tight_layout=True)
    for i, t in enumerate(ts):
        x = np.abs(img_c2.copy())
        x[x < np.percentile(x, t)] = 0
        plt.subplot(1, len(ts), i + 1)
        plt.imshow(np.log(1 + x), 'gray', interpolation='none', extent=[-h, h, -w, w])
        plt.gca().set_title(f'Low Pass by Percentile: {t}%')
    plt.savefig(f'det-lowpass-{tag}.{attack}.pdf')


if __name__ == '__main__':
    set_ccs_font(12)
    plot_low_pass('none')
    plot_low_pass('bad')
    plot_low_pass('median')
    plot_low_pass('random')
    plot_all()
