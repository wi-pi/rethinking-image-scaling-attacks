import os

"""Run SimBA on small images.
"""
for i in range(8):
    cmd = f'nohup python -u -m exp.attack_all_simba' \
          f' -l {i} -r 233 -s 8 -g 3 --query 20000' \
          f' 2>&1 > static/log/simba_small_{i}.log &'
    os.system(cmd)


"""Old
"""
# for i in range(8):
#    cmd = f'nohup python -m scripts.blackbox ' \
#          f'--scale 3 --defense median -l {i} -r 100 -s 8 -g {i} ' \
#          f'2>&1 > static/log/bb_med19_{i}.log &'
#    os.system(cmd)
