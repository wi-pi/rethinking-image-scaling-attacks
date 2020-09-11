import os
import numpy as np

from scaleadv.datasets.imagenet import create_dataset
from scaleadv.attacks.scale import ScaleAttack, SuppScalingLibraries, SuppScalingAlgorithms
from scaling.ScalingGenerator import ScalingGenerator


def get_src(data, index):
    img, y = data[index]
    x = np.array(img)
    return x

def get_tgt(data, index):
    img, y = data[index]
    img = img.resize((256, 256))
    #mg = img.resize((224, 224))
    x = np.array(img)
    return x

def save(x, name, path=''):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 3 and x.shape[-1] == 3
    assert x.min() >= 0 and x.max() > 1 and x.max() <= 255

    from PIL import Image
    img = Image.fromarray(x.astype(np.uint8))
    filename = os.path.join(path, f'{name}.pdf')
    img.save(filename)
    print(f'Saving "i{filename}".')


if __name__ == '__main__':
    data = create_dataset('static/datasets/imagenet/val/', transform=None)
    path = 'static/results/scale/'
    os.makedirs(path, exist_ok=True)
    src_id = 1000
    tgt_id = 3000

    # get data
    src = get_src(data, src_id)
    tgt = get_tgt(data, tgt_id)

    # scale attack
    attack = ScaleAttack(lib=SuppScalingLibraries.PIL, algo=SuppScalingAlgorithms.NEAREST)
    scale = ScalingGenerator.create_scaling_approach(src.shape, tgt.shape, attack.lib, attack.algo)
    imgs = x_scl, x_scl_protected, x_ada, x_ada_protected = attack.generate(src, tgt)

    # save imgs
    caps = 'x_scl', 'x_scl_protected', 'x_ada', 'x_ada_protected'
    for x, n in zip(imgs, caps):
        save(x, f'{src_id}_{tgt_id}.{n}.big', path)
        save(scale.scale_image(x), f'{src_id}_{tgt_id}.{n}.small', path)
    save(src, f'{src_id}_{tgt_id}.src', path)
    save(tgt, f'{src_id}_{tgt_id}.tgt', path)

