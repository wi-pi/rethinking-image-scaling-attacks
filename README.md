# Scale-Adv

Understanding the real potential of Image-Scaling attacks.

## Install

* Clone this repository with `git clone --recursive git@github.com:wi-pi/Scale-Adv.git`
* Install PyTorch and run `pip install -f requirements.txt`

## Prerequisites

* Download ImageNet dataset (validation) in `static/datasets/imagenet/val`
* Download [robust models](https://github.com/MadryLab/robustness#pretrained-models) in `static/models/`
* Define your datasets like `scaleadv/datasets/imagenet.py`
* Define your models like `scaleadv/models/resnet.py`

## Common Usage Examples

### Hide a given adversarial example

> See detailed help in `python -m scaleadv.tests.scale_adv -h`

**Example: Hide PGD-30 with L2-20 to bypass the median defense.**

```sh
python -m scaleadv.tests.scale_adv \
--id 5000 --target 200 --robust 2 \
--lib cv --algo linear --bigger 3 \
--eps 20 --step 30 \
--lr 0.01 --lam-inp 8 --iter 1000 --defense median --mode sample
```

**Example: Hide PGD-30 with L2-20 to bypass the random defense.**

```sh
python -m scaleadv.tests.scale_adv \
--id 5000 --target 200 --robust 2 \
--lib cv --algo linear --bigger 3 \
--eps 20 --step 30 \
--lr 0.1 --lam-inp 200 --iter 120 --defense random --mode sample
```

### Generate an HR adversarial example

> See detailed help in `python -m scaleadv.tests.scale_adv_optimal -h`

**Example: Generate HR adv-example with PGD-100 and L2-50 to bypass median defense.**

```sh
python -m scaleadv.tests.scale_adv_optimal \
--id 5000 --target 200 --robust 2 \
--lib cv --algo linear --bigger 3 \
--eps 20 --step 30 \
--big-eps 50 --big-step 100 \
--defense median
```

**Example: Generate HR adv-example with PGD-100 and L2-50 to bypass random defense.**

```sh
python -m scaleadv.tests.scale_adv_optimal \
--id 5000 --target 200 --robust 2 \
--lib cv --algo linear --bigger 3 \
--eps 20 --step 30 \
--big-eps 50 --big-step 100 \
--defense random
```

**Example: Bypass random defense with Cheap/Laplacian approximation.**

```sh
python -m scaleadv.tests.scale_adv_optimal \
--id 5000 --target 200 --robust 2 \
--lib cv --algo linear --bigger 3 \
--eps 20 --step 30 \
--big-eps 50 --big-step 100 \
--defense cheap/laplace
```

### Plot random pooling's histogram.

Add `--lapace` to plot Laplacian approximations.

```sh
python -m scaleadv.experiments.random_hist \
--id 5000 \
--lib cv --algo linear \
--laplace
```

## Acknowledgements

* Pretrained Robust Models
  * https://github.com/MadryLab/robustness#pretrained-models
* Previous Image-Scaling Attack's Implementation
  * https://github.com/yfchen1994/scaling_camouflage
  * https://github.com/EQuiw/2019-scalingattack/tree/master/scaleatt
