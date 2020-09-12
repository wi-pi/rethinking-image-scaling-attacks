#!/bin/bash
# Test scaleadv for 51 images from ImageNet.
CMD="python -m scaleadv.tests.scaleadv"
LOG="static/results/scaleadv"

for id in `seq 0 1000 50000`; do
    cmd="$CMD $id 2>&1 | tee $LOG/$id.log"
    echo $cmd && eval $cmd
done
