import sys
import os
import yaml

if len(sys.argv) > 1:
    GPU_ID = str(sys.argv[1])
else:
    GPU_ID = ''
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

sys.path.append('..')
from seq2tens.datasets import get_available_datasets

from sacred.observers import FileStorageObserver

from experiment import ex

num_repetitions = 10
tmp_dir = './tmp/'
save_dir = './benchmarks/'

ex.observers.append(FileStorageObserver(save_dir))

# load dataset specs
datasets = get_available_datasets()

# load models
with open('./configs/configs.yaml', 'r') as f:
    configs = list(yaml.load(f, Loader=yaml.SafeLoader).keys())

if not os.path.isdir(tmp_dir):
    os.mkdir(tmp_dir)
    
for i in range(num_repetitions):
    for config_name in configs:
        for dataset_name in datasets:
            exp_name = f'{config_name}_{dataset_name}_{i}'
            exp_fp = os.path.join(tmp_dir, exp_name + '.txt')
            if os.path.exists(exp_fp):
                print(f'Information | benchmarks.py : tmp file for experiment {exp_name} already exists, skipping to the next.')
                continue
            with open(exp_fp, 'w') as f:
                pass
            ex.run(named_configs=[config_name, 'dataset.' + dataset_name])