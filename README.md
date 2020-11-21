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

### Set global arguments

> See detailed help in `python -m scaleadv.tests.scale_adv -h`
>
> These arguments are generally the `...` parts in Hide/Generate examples.

**Example: Test cv.linear on PGD-30 with L2-20 to bypass the median defense.**

```sh
python -m sclaeadv.tests.scale_adv \
--id 5000 --target 200 --model 2 \
--lib cv --algo linear --scale 3 \
--eps 20 --step 30 \
--defense median
```

**Example: Test cv.linear on PGD-30 with L2-20 to bypass the random defense, approximate by cheap sampling.**

```sh
python -m sclaeadv.tests.scale_adv \
--id 5000 --target 200 --model 2 \
--lib cv --algo linear --scale 3 \
--eps 20 --step 30 \
--defense random --mode cheap --samples 200
```

### Hide a given adversarial example

> See detailed help in `python -m scaleadv.tests.scale_adv hide -h`

**Example: Hide to bypass the median defense.**

```sh
python -m scaleadv.tests.scale_adv ... \
--defense median \
hide --lr 0.01 --lam-inp 8 --iter 1000
```

**Example: Hide to bypass the random defense.**

```sh
python -m scaleadv.tests.scale_adv ... \
--defense random --mode cheap --samples 200 \
hide --lr 0.1 --lam-inp 200 --iter 120
```

### Generate an HR adversarial example

> See detailed help in `python -m scaleadv.tests.scale_adv generate -h`

**Example: Generate to bypass the median defense.**

```sh
python -m scaleadv.tests.scale_adv ... \
--defense median \
generate --big-eps 40 --big-sig 4.0 --big-step 100
```

**Example: Generate to bypass the random defense.**

```sh
python -m scaleadv.tests.scale_adv ... \
--defense random --mode cheap --samples 200 \
generate --big-eps 40 --big-sig 4.0 --big-step 100
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
* Pretrained Smoothing Models
  * https://github.com/locuslab/smoothing#getting-started
  * https://github.com/Hadisalman/smoothing-adversarial#download-our-pretrained-models
* Previous Image-Scaling Attack's Implementation
  * https://github.com/yfchen1994/scaling_camouflage
  * https://github.com/EQuiw/2019-scalingattack/tree/master/scaleatt
