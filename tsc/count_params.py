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

from params import ex

save_dir = './params/'

ex.observers.append(FileStorageObserver(save_dir))

# load dataset specs
datasets = get_available_datasets()

# load models
with open('./configs.yaml', 'r') as f:
    configs = list(yaml.load(f, Loader=yaml.SafeLoader).keys())

for config_name in configs:
    for dataset_name in datasets:
        ex.run(named_configs=[config_name, 'dataset.' + dataset_name])