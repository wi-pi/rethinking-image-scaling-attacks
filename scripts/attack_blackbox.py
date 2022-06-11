import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from art.estimators.classification import PyTorchClassifier, BlackBoxClassifier
from loguru import logger
from torch import nn

from src.attacks.hop_skip_jump import HSJ
from src.attacks.sign_opt import SignOPT
from src.attacks.smart_noise import SmartNoise
from src.datasets.utils import DatasetHelper
from src.defenses import POOLING_MAPS
from src.models import ScalingLayer, imagenet_resnet50, celeba_resnet34
from src.models.online import OnlineModel
from src.scaling import str_to_lib, str_to_alg, ScalingAPI


def main(args):
    # Basic
    base = 224
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
    output_path = args.output / f'{args.dataset}_{args.model}_{args.attack}_{args.defense}_{args.scale}x' / args.tag
    logger.info(f'Save results to "{output_path}".')
    os.makedirs(output_path, exist_ok=True)

    # Load dataset
    dataset = DatasetHelper(name=args.dataset, scale=args.scale, base=base, valid_samples_only=True)
    if args.num_eval:
        args.id = dataset.sample(args.num_eval)

    # Load scaling
    lr_size, hr_size = base, base * args.scale
    lr_shape, hr_shape = (3, lr_size, lr_size), (3, hr_size, hr_size)
    scaling = ScalingAPI(hr_shape[1:], lr_shape[1:], args.lib, args.alg)

    # Load pooling layer (exact)
    pooling_layer = None
    if args.defense != 'none':
        pooling_layer = POOLING_MAPS[args.defense].from_api(scaling)

    # Load pooling layer (estimate)
    pooling_layer_estimate = pooling_layer
    if args.defense == 'median' and not args.no_smart_median:
        pooling_layer_estimate = POOLING_MAPS['quantile'].like(pooling_layer)

    # Load scaling layer
    scaling_layer = None
    if args.scale > 1:
        scaling_layer = ScalingLayer.from_api(scaling).eval().cuda()

    # Synthesize projection (only combine non-None layers)
    projection = nn.Sequential(*filter(None, [pooling_layer, scaling_layer]))
    projection_estimate = nn.Sequential(*filter(None, [pooling_layer_estimate, scaling_layer]))

    # Load class network & end-to-end ART classifier
    match args.model:
        case 'imagenet':
            lr_network = imagenet_resnet50('nature', normalize=True)
            hr_network = nn.Sequential(projection, lr_network)
            classifier = PyTorchClassifier(hr_network, nn.CrossEntropyLoss(), hr_shape, 1000, clip_values=(0, 1))
        case 'celeba':
            lr_network = celeba_resnet34(num_classes=11, binary_label=6, ckpt='nature')
            hr_network = nn.Sequential(projection, lr_network)
            classifier = PyTorchClassifier(hr_network, nn.CrossEntropyLoss(), hr_shape, 2, clip_values=(0, 1))
        case 'api':
            # blackbox contains an internal scaling layer, which is the `scaling_layer` as we inferred.
            hr_network = OnlineModel()
            classifier = BlackBoxClassifier(hr_network.predict, hr_shape, 2, clip_values=(0, 1))
        case _:
            raise NotImplementedError(f'Unknown model "{args.model}".')

    # Load Smart Noise
    smart_noise = None
    if args.scale > 1 and not args.no_smart_noise:
        smart_noise = SmartNoise(
            hr_shape=(3, hr_size, hr_size),
            lr_shape=(3, lr_size, lr_size),
            projection=projection,
            projection_estimate=projection_estimate,
            precise=args.precise_noise,
        )

    # Attack loop
    for i in args.id:
        # Load data
        x, y = dataset[i]
        logger.info(f'Loading source image: id {i}, label {y}, shape {x.shape}, dtype {x.dtype}.')

        # Special process for api attack
        if args.model == 'api':
            try:
                api_label = hr_network.set_current_sample(x)
                logger.info(f'Retrieving API label: {api_label}')
                if api_label is None:
                    continue
            except Exception as exc:
                logger.error(exc)

        # Load attack (within loop since the attack could be stateful, which we want to avoid)
        match args.attack:
            case 'hsj':
                attack = HSJ(classifier, max_iter=150, max_eval=200, max_query=args.query, smart_noise=smart_noise)
            case 'opt':
                attack = SignOPT(classifier, max_iter=1000, max_query=args.query, smart_noise=smart_noise)
            case _:
                raise NotImplementedError(f'Unknown attack "{args.attack}".')

        # Attack
        try:
            attack.generate(x[None], np.array([y]))
        except Exception as exc:
            logger.error(exc)

        # Dump logs
        if attack.log:
            df = pd.DataFrame(attack.log, columns=['Query', 'Perturbation'])
            df['Perturbation'] /= args.scale
            df.to_csv(output_path / f'{i}.csv', index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    _ = parser.add_argument
    # Test Samples
    _('-i', '--id', type=int, nargs='+', help='test image IDs')
    _('-n', '--num-eval', type=int, help='number of randomly sampled images to test')
    # Dataset & Model
    _('-d', '--dataset', type=str, choices=['imagenet', 'celeba'], required=True, help='test dataset')
    _('-m', '--model', type=str, choices=['imagenet', 'celeba', 'api'], help='test model')
    # Scaling
    _('--lib', default='cv', type=str, choices=str_to_lib, help='scaling libraries')
    _('--alg', default='linear', type=str, choices=str_to_alg, help='scaling algorithms')
    _('-s', '--scale', default=None, type=int, help='set a fixed scale ratio, unset to use the original size')
    # Defense
    _('--defense', default='none', choices=POOLING_MAPS, help='type of defense')
    # Attack
    _('-a', '--attack', default='hsj', choices=['hsj', 'opt'], help='attack type')
    _('-q', '--query', default=25000, type=int, help='query budget')
    _('--no-smart-noise', action='store_true', help='disable scaling-aware noise sampling')
    _('--no-smart-median', action='store_true', help='disable gradient-efficient median approximation')
    _('--precise-noise', action='store_true', help='use the straightforward scaling-aware noise sampling')
    # Misc
    _('-o', '--output', type=Path, default='static/logs', help='path to output directory')
    _('-t', '--tag', default='', type=str, help='name for this experiment')
    _('-g', '--gpu', default=0, type=int, help='GPU id')
    args = parser.parse_args()

    """
    Sanity check for args
    """
    # Scale=1 should not have preprocessing
    if args.scale == 1:
        assert args.defense == 'none'
        assert args.no_smart_noise is False
        assert args.no_smart_median is False
        assert args.precise_noise is False

    # API should not have defense
    if args.model == 'api':
        assert args.defense == 'none'
        assert args.query <= 3000

    # Ad-hoc fix for default args
    if args.model is None:
        args.model = args.dataset

    return args


if __name__ == '__main__':
    main(parse_args())
