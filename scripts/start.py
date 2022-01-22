import os
from itertools import cycle

"""Hide SimBA's small adv to large original (without defense).
"""
# n_job = 12
# for i, gpu in zip(range(n_job), cycle([0, 1, 2, 3])):
#     cmd = f'nohup python -u -m exp.attack_all_simba_hide' \
#           f' -l {i} -r 233 -s {n_job} -g {gpu}' \
#           f' 2>&1 > static/log/simba_hide_{i}.log &'
#     os.system(cmd)

"""Run SimBA on small images.
"""
# n_job = 12
# for i, gpu in zip(range(n_job), cycle([0, 1, 2, 3])):
#     cmd = f'nohup python -u -m exp.attack_all_simba_small' \
#           f' -l {i} -r 233 -s {n_job} -g {gpu} --query 20000' \
#           f' 2>&1 > static/log/simba_small_{i}.log &'
#     os.system(cmd)


"""Old
"""
# for i in range(8):
#    cmd = f'nohup python -m scripts.blackbox ' \
#          f'--scale 3 --defense median -l {i} -r 100 -s 8 -g {i} ' \
#          f'2>&1 > static/log/bb_med19_{i}.log &'
#    os.system(cmd)


"""TODO: Run HSJA (no defense) without smart noise.
"""
# n_job = 4
# gpus = [0, 1, 2, 3]
# for i, gpu in zip(range(n_job), cycle(gpus)):
#     cmd = f'nohup python -m scripts.blackbox' \
#           f' --scale 3 --defense none -l {i} -r 100 -s {n_job} -g {gpu} --no-smart-noise --tag badnoise' \
#           f' 2>&1 > static/log/bb_none_bad_{i}.log &'
#     os.system(cmd)


"""TODO: Run HSJA (median defense) without smart median.
"""
# n_job = 4
# gpus = [0, 1, 2, 3]
# for i, gpu in zip(range(n_job), cycle(gpus)):
#     cmd = f'nohup python -m scripts.blackbox' \
#           f' --scale 3 --defense median -l {i} -r 100 -s {n_job} -g {gpu} --no-smart-median --tag badmedian' \
#           f' 2>&1 > static/log/bb_median_bad_{i}.log &'
#     os.system(cmd)


"""Run HSJA on small images.
"""
# n_job = 4
# gpus = [0, 1, 2, 3]
# for i, gpu in zip(range(n_job), cycle(gpus)):
#     cmd = f'nohup python -m exp.attack_all_hsj_small' \
#           f' --scale 3 -l {i} -r 100 -s {n_job} -g {gpu} --tag small' \
#           f' 2>&1 > static/log/bb_small_{i}.log &'
#     os.system(cmd)


"""Run SignOPT on small images. [RUNNING guard]
"""
n_job = 4
gpus = [0, 1, 2, 3]
for i, gpu in zip(range(n_job), cycle(gpus)):
    cmd = f'nohup python -m exp.attack_all_hsj_small' \
          f' --scale 3 -l {i} -r 100 -s {n_job} -g {gpu} --tag opt_small --attack opt' \
          f' 2>&1 > static/log/bb_small_opt_{i}.log &'
    os.system(cmd)


"""Run SignOPT (no defense) with smart noise [RUNNING guard]
"""
# n_job = 4
# gpus = [0, 1, 2, 3]
# for i, gpu in zip(range(n_job), cycle(gpus)):
#     cmd = f'nohup python -m scripts.blackbox' \
#           f' --scale 3 --defense none -l {i} -r 100 -s {n_job} -g {gpu} --tag opt_good --attack opt' \
#           f' 2>&1 > static/log/bb_none_opt_good_{i}.log &'
#     os.system(cmd)


"""Run SignOPT (median defense) with smart noise [RUNNING guard]
"""
# n_job = 4
# gpus = [0, 1, 2, 3]
# for i, gpu in zip(range(n_job), cycle(gpus)):
#     cmd = f'nohup python -m scripts.blackbox' \
#           f' --scale 3 --defense median -l {i} -r 100 -s {n_job} -g {gpu} --tag opt_good --attack opt' \
#           f' 2>&1 > static/log/bb_median_opt_good_{i}.log &'
#     os.system(cmd)
