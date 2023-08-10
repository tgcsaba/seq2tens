import os
import yaml
import random

from sklearn.model_selection import ParameterGrid

from warnings import warn

from copy import deepcopy

num_samples = 20

with open('./configs/configs_default.yaml', 'r') as f:
    defaults = yaml.load(f, Loader=yaml.SafeLoader)
    
with open('./configs/search_spaces.yaml', 'r') as f:
    spaces = yaml.load(f, Loader=yaml.SafeLoader)
    
config_names = list(defaults.keys())

def _set_cfg_value(_cfg, _entry, _value): 
    _entry_split = _entry.split('.')
    for k in _entry_split[:-1]:
        _cfg = _cfg[k]
    _cfg[_entry_split[-1]] = _value

random_configs = {}
for cfg_name in config_names:
    if cfg_name not in spaces:
        warn(f'Warning | gen_random_configs.py: No search space defined for {cfg_name}; skipping.')
        continue
    search_space = list(ParameterGrid(spaces[cfg_name]))
    samples = random.sample(search_space, num_samples)
    for i, sample in enumerate(samples):
        rand_cfg = deepcopy(defaults[cfg_name])
        for entry, value in sample.items():
            _set_cfg_value(rand_cfg, entry, value)
        rand_cfg_name = cfg_name + str(i+1)
        random_configs[rand_cfg_name] = rand_cfg

        
if os.path.exists('./configs/configs_random.yaml'):
    os.remove('./configs/configs_random.yaml')
with open('./configs/configs_random.yaml', 'w') as f:
    yaml.dump(random_configs, f)