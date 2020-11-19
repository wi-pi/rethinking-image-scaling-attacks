"""
This module implements test APIs for Scale-Adv Attack (L2) based on off-the-shelf adv attacks.

Notes:
    1. Adv-attack on large image may require more steps/budges, like PGD-100 and eps * sqrt(scale-ratio).
    2. `average` is fast, but may not work for random-defense, unless the budget has equal PSNR.
    3. `random` is slow, but can bypass random-defense even if the budget is eps * 2.
    4. `laplace` is laplacian distribution centered at `average`, it can compensate (a little) its issue, with more steps.

Empirical good settings:
    1. 4 * 40 / 30 eps_step with 300 steps
"""
from scaleadv.attacks.utils import estimate_mad
from scaleadv.models.scaling import FullScaleNet
from scaleadv.models.utils import AverageGradientClassifier, ReducedCrossEntropyLoss
from scaleadv.tests.scale_adv import *

if __name__ == '__main__':
    p = ArgumentParser()
    # Input args
    p.add_argument('--id', type=int, required=True, help='ID of test image')
    p.add_argument('--target', type=int, required=True, help='target label')
    p.add_argument('--robust', default=None, type=str, choices=ROBUST_MODELS, help='use robust model, optional')
    # Scaling args
    p.add_argument('--lib', default='cv', type=str, choices=LIB, help='scaling libraries')
    p.add_argument('--algo', default='linear', type=str, choices=ALGO, help='scaling algorithms')
    p.add_argument('--bigger', default=1, type=int, help='scale up the source image')
    # Adversarial attack args
    p.add_argument('--eps', default=20, type=float, help='L2 perturbation of adv-example')
    p.add_argument('--big-eps', default=60, type=float, help='L2 perturbation of attack image')
    p.add_argument('--step', default=30, type=int, help='max iterations of PGD attack')
    p.add_argument('--big-step', default=30, type=int, help='max iterations of Scale-Adv')
    p.add_argument('--adv-proxy', action='store_true', help='do adv-attack on noisy proxy')
    # Scaling attack args
    p.add_argument('--defense', default=None, type=str, choices=POOLING.keys(), help='type of defense')
    args = p.parse_args()

    # Load data
    dataset = create_dataset(transform=None)
    src, _ = dataset[args.id]
    src = resize_to_224x(src, more=args.bigger, square=True)
    src = np.array(src)

    # Load scaling
    lib = LIB_TYPE[args.lib]
    algo = ALGO_TYPE[args.algo]
    scaling = ScalingGenerator.create_scaling_approach(src.shape, INPUT_SHAPE_PIL, lib, algo)
    mask = get_mask_from_cl_cr(scaling.cl_matrix, scaling.cr_matrix)

    # Convert data to batch ndarray
    normalize_to_batch = T.Compose([T.ToTensor(), lambda x: x.numpy()[None, ...]])
    src_inp = scaling.scale_image(src)
    src, src_inp = map(normalize_to_batch, [src, src_inp])

    # Compute scale ratio
    sr_h, sr_w = [src.shape[i] // src_inp.shape[i] for i in [2, 3]]

    # Load pooling
    # TODO: Support non-square pooling
    pooling = NonePool2d()
    k = sr_h * 2 - 1
    pooling_args = (k, 1, k // 2, mask)
    nb_samples = NUM_SAMPLES_SAMPLE if args.defense in ['random', 'laplace', 'cheap'] else 1
    if args.defense:
        pooling = POOLING[args.defense](*pooling_args)
    if args.defense == 'laplace':
        mad = estimate_mad(src, k) * 2.0
        pooling.update_dist(scale=mad)

    # Load networks
    scale_net = ScaleNet(scaling.cl_matrix, scaling.cr_matrix).eval()
    class_net = nn.Sequential(NormalizationLayer.from_preset('imagenet'), resnet50_imagenet(args.robust)).eval()
    if nb_samples > 1:
        class_net = BalancedDataParallel(FIRST_GPU_BATCH, class_net)

    # Move networks to GPU
    scale_net = scale_net.cuda()
    class_net = class_net.cuda()

    # Adv-Attack on src_inp
    y_target = np.eye(NUM_CLASSES, dtype=np.int)[None, args.target]
    classifier = PyTorchClassifier(class_net, nn.CrossEntropyLoss(), INPUT_SHAPE_NP, NUM_CLASSES, clip_values=(0, 1))
    eps = args.eps
    eps_step = 2.5 * eps / args.step
    adv_attack = IndirectPGD(classifier, 2, eps, eps_step, args.step, targeted=True, batch_size=NUM_SAMPLES_PROXY)
    adv = adv_attack.generate(x=src_inp, y=y_target, proxy=None)

    # Adv-Attack on src
    full_net = FullScaleNet(scale_net, class_net, pooling, n=1)
    y_src = classifier.predict(src_inp).argmax(1).item()
    new_args = dict(nb_samples=nb_samples, verbose=True, y_cmp=[y_src, args.target])
    classifier = AverageGradientClassifier(full_net, ReducedCrossEntropyLoss(), tuple(src.shape[1:]), NUM_CLASSES,
                                           **new_args, clip_values=(0, 1))
    eps = args.big_eps
    eps_step = 4 * eps / args.step
    adv_attack = IndirectPGD(classifier, 2, eps, eps_step, args.big_step, targeted=True, batch_size=NUM_SAMPLES_PROXY)
    att = adv_attack.generate(x=src, y=y_target, proxy=None)

    # Test
    e = Evaluator(scale_net, class_net, pooling_args)
    e.eval(src, adv, att, summary=True, tag=f'{args.id}.{args.defense}.optimal.eps{args.eps}', save='.',
           y_adv=args.target)
