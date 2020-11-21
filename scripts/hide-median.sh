#!/bin/sh
# Input args
ID=5000
TARGET=200
MODEL=2
TAG="TEST"

# ScaleAdv args
LR=0.01
LAM_INP=7
ITER=1000

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
  hide \
  --lr $LR --lam-inp $LAM_INP --iter $ITER

