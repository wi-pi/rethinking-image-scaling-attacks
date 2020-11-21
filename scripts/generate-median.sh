#!/bin/sh
# Input args
ID=5000
TARGET=200
MODEL=2
TAG="TEST"

# ScaleAdv args
BIG_EPS=40
BIG_SIG=4.0
BIG_STEP=120

# Scaling args
LIB=cv
ALGO=linear
SCALE=3

# Adv args
EPS=20
STEP=30

python -m scaleadv.tests.scale_adv \
  --id $ID --target $TARGET --model $MODEL \
  --lib $LIB --algo $ALGO --scale $SCALE \
  --eps $EPS --step $STEP \
  --defense median \
  --tag $TAG \
  generate \
  --big-eps $BIG_EPS --big-sig $BIG_SIG --big-step $BIG_STEP

