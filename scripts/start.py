import os

for i in range(8):
    cmd = f'nohup python -m scripts.blackbox ' \
          f'--scale 3 --defense median -l {i} -r 100 -s 8 -g {i} ' \
          f'2>&1 > static/log/bb_med19_{i}.log &'
    os.system(cmd)