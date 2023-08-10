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
save_dir = './random_search/'

ex.observers.append(FileStorageObserver(save_dir))

# load model configs
if not os.path.exists('./configs/configs_random.yaml'):
    raise RuntimeError('Error | run_random_search.py : missing file \'./configs/configs_random.yaml\'.\
                       Please run the  \'./gen_random_configs.py\' script first.')
with open('./configs/configs_random.yaml', 'r') as f:
    configs = list(yaml.load(f, Loader=yaml.SafeLoader).keys())

if not os.path.isdir(tmp_dir):
    os.mkdir(tmp_dir)
    
for config_name in configs:
    tmp_fp = os.path.join(tmp_dir, config_name + '.txt')
    if os.path.exists(tmp_fp):
        print(f'Information | run_random_search.py : tmp file for experiment {config_name} already exists, skipping to the next.')
        continue
    with open(tmp_fp, 'w') as f:
        pass
    ex.run(named_configs=[config_name])