from scaleadv.tests.psnr_vs_psnr import *
from scaleadv.tests.scale_adv_optimal import *


class MetricCompare_Optimal(MetricCompare):
    STEP = 100

    def eval_lib_algo(self, lib: str, algo: str, defense: str = None, dump=True, load=False):
        fname = f'{self.tag}.{lib}.{algo}.{defense}.optimal.pkl'
        if load and os.path.exists(fname):
            return pickle.load(open(fname, 'rb'))

        # Load scaling algorthm
        lib_t, algo_t = LIB_TYPE[lib], ALGO_TYPE[algo]
        scaling = ScalingGenerator.create_scaling_approach(self.src.shape, INPUT_SHAPE_PIL, lib_t, algo_t)
        mask = get_mask_from_cl_cr(scaling.cl_matrix, scaling.cr_matrix)

        # Get src as batch ndarray
        inp = scaling.scale_image(self.src)
        src, inp = map(self.pil_to_batch, [self.src, inp])

        # Get pooling layer
        pooling = NonePool2d()
        kernel = src.shape[2] // inp.shape[2] * 2 - 1
        pooling_args = kernel, 1, kernel // 2, mask
        nb_samples = NUM_SAMPLES_SAMPLE if defense in ['random', 'laplace', 'cheap'] else 1
        if defense not in [None, 'none']:
            pooling = POOLING[defense](*pooling_args)
        if defense == 'laplace':
            mad = estimate_mad(src, kernel) * 2.0
            pooling.update_dist(scale=mad)

        # Get networks
        scale_net = ScaleNet(scaling.cl_matrix, scaling.cr_matrix).eval()
        class_net = self.class_net.eval()
        if nb_samples > 1:
            class_net = BalancedDataParallel(FIRST_GPU_BATCH, class_net)
        scale_net = scale_net.cuda()
        class_net = class_net.cuda()
        full_net = FullScaleNet(scale_net, class_net, pooling, n=1)

        # Get art's proxy
        y = np.eye(NUM_CLASSES, dtype=int)[None, self.target]
        new_args = dict(nb_samples=nb_samples, verbose=True, y_cmp=[100, args.target])  # TODO: avoid hard-coded label
        classifier = AverageGradientClassifier(full_net, ReducedCrossEntropyLoss(), tuple(src.shape[1:]), NUM_CLASSES,
                                               **new_args, clip_values=(0, 1))

        data = []
        for eps in self.eps:
            # Get att
            eps_step = 4 * eps / 30
            adv_attack = IndirectPGD(classifier, 2, eps, eps_step, self.STEP, targeted=True,
                                     batch_size=NUM_SAMPLES_PROXY)
            att = adv_attack.generate(x=src, y=y)

            # Get fake adv
            adv = inp

            # Test
            e = Evaluator(scale_net, class_net, pooling_args)
            stats = e.eval(src, adv, att, summary=True, tag=f'EVAL_OPT.{lib}.{algo}.{defense}.eps{eps}', save='.',
                           y_adv=self.target)
            data.append(stats)

        if dump:
            pickle.dump(data, open(fname, 'wb'))
        return data

    def plot_lib_algo(self, lib: str, algo: str, defense: str, marker: str):
        data = self.eval_lib_algo(lib, algo, defense, dump=True, load=True)
        y_label = {'none': 'Y', 'median': 'Y_MED'}.get(defense, 'Y_RND')
        tag = 'random' if defense in ['laplace', 'cheap'] else defense
        x, y = [], []
        warning = []
        for d in data:
            att_ok = np.mean(d['ATT'][y_label] == args.target) > 0.7
            att_score = d['ATT']['L-2'].cpu().item() / 3
            adv_score = d[f'ATT-INP ({tag})']['L-2'].cpu().item()
            if att_ok:
                x.append(att_score)
                y.append(adv_score)

        if x:
            plt.plot(x, y, marker, lw=1, ms=1.75, label=f'{lib}.{algo} ({tag})')
        if warning:
            plt.scatter(*zip(*warning), s=2, c='k', marker='o')


if __name__ == '__main__':
    p = ArgumentParser()
    # Input args
    p.add_argument('--id', type=int, required=True, help='ID of test image')
    p.add_argument('--target', type=int, required=True, help='target label')
    p.add_argument('--robust', default=None, type=str, choices=ROBUST_MODELS, help='use robust model, optional')
    # Scaling args
    p.add_argument('--bigger', default=1, type=int, help='scale up the source image')
    # Adversarial attack args
    p.add_argument('--eps', default=20, type=float, help='L2 perturbation of adv-example')
    p.add_argument('--step', default=30, type=int, help='max iterations of PGD attack')
    p.add_argument('--adv-proxy', action='store_true', help='do adv-attack on noisy proxy')
    # Scaling attack args
    p.add_argument('--lr', default=0.01, type=float, help='learning rate for scaling attack')
    p.add_argument('--lam-inp', default=1, type=int, help='lambda for L2 penalty at the input space')
    p.add_argument('--lam-ce', default=2, type=int, help='lambda for CE penalty')
    p.add_argument('--iter', default=200, type=int, help='max iterations of Scaling attack')
    p.add_argument('--defense', default=None, type=str, choices=POOLING.keys(), help='type of defense')
    p.add_argument('--mode', default='none', type=str, choices=ADAPTIVE_MODE, help='adaptive attack mode')
    args = p.parse_args()

    # Load data
    dataset = create_dataset(transform=None)
    src, _ = dataset[args.id]
    src = resize_to_224x(src, more=args.bigger)
    src = np.array(src)
    ratio = src.shape[0] // 224

    # Load networks
    class_net = nn.Sequential(NormalizationLayer.from_preset('imagenet'), resnet50_imagenet(args.robust)).eval()

    # Other params
    # eps = list(range(5, 51, 5))
    eps = list(range(30, 121, 5))
    lib = 'cv'
    algo_list = ['linear']
    defense_list = ['none', 'median', 'cheap']
    color = 'o- o--'.split()
    marker = 'C2 C0 C1'.split()

    # Tester
    tester = MetricCompare_Optimal(class_net, src, args.target, eps, tag=f'eval.{args.id}')

    # Test + Plot
    plt.figure(figsize=(5, 5), constrained_layout=True)
    plt.title(f'Scale-Adv Attack (Generate)')
    plt.xlabel('Scale-Adv Attack (L-2)')
    plt.ylabel('Adv Example after Defense and Scaling (L-2)')
    for algo, c in zip(algo_list, color):
        for defense, m in zip(defense_list, marker):
            tester.plot_lib_algo(lib, algo, defense, marker=f'{m}{c}')

    # Plot reference line
    mi, ma = eps[0] // 3, eps[-1] // 3
    plt.plot([mi, ma], [mi, ma], 'k--', lw=1, label='reference')
    plt.legend(loc='upper left')
    plt.savefig(f'test.pdf')
