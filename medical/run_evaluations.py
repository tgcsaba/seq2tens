import sys
import os
import yaml

if len(sys.argv) > 1:
    GPU_ID = str(sys.argv[1])
else:
    GPU_ID = ''
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID


from sacred.observers import FileStorageObserver

from experiment import ex

tmp_dir = './tmp/'
save_dir = './evaluations/'

num_repetitions = 5

ex.observers.append(FileStorageObserver(save_dir))

# load best configs
if not os.path.exists('./configs/configs_best.yaml'):
    raise RuntimeError('Error | run_evaluations.py : missing file \'./configs/configs_best.yaml\'.\
                       Please run the  \'./gen_best_configs.py\' script first.')
with open('./configs/configs_best.yaml', 'r') as f:
    configs = list(yaml.load(f, Loader=yaml.SafeLoader).keys())

if not os.path.isdir(tmp_dir):
    os.mkdir(tmp_dir)

for i in range(num_repetitions):
    for config_name in configs:
        tmp_fp = os.path.join(tmp_dir, f'{config_name}2_{i}.txt')
        if os.path.exists(tmp_fp):
            print(f'Information | evaluations.py : tmp file for experiment {config_name} already exists, skipping to the next.')
            continue
        with open(tmp_fp, 'w') as f:
            pass
        ex.run(named_configs=[config_name])