#!/bin/sh
# Input args
ID=5000
TARGET=200
MODEL=2
TAG="TEST"

# ScaleAdv args
LR=0.1
LAM_INP=100
ITER=150
MODE=sample
SAMPLES=200

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
  --defense random --mode $MODE --samples $SAMPLES \
  --tag $TAG \
  hide \
  --lr $LR --lam-inp $LAM_INP --iter $ITER

