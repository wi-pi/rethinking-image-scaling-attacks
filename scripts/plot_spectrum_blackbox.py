import cv2
import matplotlib.pyplot as plt
import numpy as np

from scaleadv.utils import set_ccs_font


def plot(name, tag):
    img_color = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
    img_spectrum = np.fft.fftshift(np.fft.fft2(cv2.imread(name, 0)))
    # save image
    h, w = img_spectrum.shape
    plt.imshow(img_color, interpolation='none', extent=[0, h, 0, w])
    plt.axis('off')
    plt.savefig(f'{tag}_image.jpg', bbox_inches='tight', pad_inches=0)
    # save spectrum
    h, w = h // 2, w // 2
    plt.imshow(np.log(1 + np.abs(img_spectrum)), 'gray', interpolation='none', extent=[-h, h, -w, w])
    plt.axis('off')
    plt.savefig(f'{tag}_spectrum.jpg', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    set_ccs_font(16)
    queries = [1, 5, 10]
    data = {
        'none': [f'blackbox-examples/bb_test.5000.none.{i:02d}.png' for i in [6, 17, 27]],
        'median': [f'blackbox-examples/bb_test.5000.median.{i:02d}.png' for i in [6, 17, 27]],
    }

    for row, (defense, imgs) in enumerate(data.items()):
        for col, name in enumerate(imgs):
            tag = f'blackbox_spectrum_{defense}_{queries[col]}K'
            plot(name, tag)
