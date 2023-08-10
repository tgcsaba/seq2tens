import os
import glob
import sys
import json
import yaml

import numpy as np

from warnings import warn

from sklearn.preprocessing import StandardScaler

from string import digits

num_samples = 20


if not os.path.isdir('./random_search/'):
    raise RuntimeError('Error | gen_best_configs.py : missing folder \'./random_search/\'.\
                       Please run the  \'./run_random_search.py\' script first.')
result_dir = './random_search/'
    
result_ps = sorted([p for p in glob.glob(os.path.join(result_dir, '*')) if os.path.basename(p) != '_sources'], key=lambda x: int(os.path.basename(x)))

cfg_to_run_id = {}
val_results = {}

for i, path in enumerate(result_ps):
    run_id = os.path.basename(path)
    met_fp = os.path.join(path, 'metrics.json')
    with open(met_fp, 'r') as f:
        mets = json.load(f)
    if 'testing.final_loss' not in mets:
        warn(f'Warning | run {run_id} has been interrupted.')
        continue
        
    cfg_fp = os.path.join(path, 'config.json')
    with open(cfg_fp, 'r') as f:
        cfg = json.load(f)
    config_name = cfg['config_name']
    
    cfg_to_run_id[config_name] = run_id
    
    val_results[config_name] = [
#         -float(mets['validation.final_loss']['values'][0]),
        float(mets['validation.final_accuracy']['values'][0]),
        float(mets['validation.final_auroc']['values'][0]),
        float(mets['validation.final_auprc']['values'][0])
    ]
    
base_names = {cfg_name.rstrip(digits) for cfg_name in val_results}
cfg_names = {base_name : [cfg_name for cfg_name in val_results if cfg_name.rstrip(digits)==base_name] for base_name in base_names}

val_metrics = {base_name : np.asarray([val_results[cfg_name] for cfg_name in val_results if cfg_name.rstrip(digits)==base_name]) for base_name in base_names}

scaler = StandardScaler()

scaler.fit(np.concatenate(list(val_metrics.values()), axis=0))

val_scores = {base_name : scaler.transform(val_metrics[base_name]) for base_name in base_names}
val_scores = {base_name : np.sum(val_scores[base_name], axis=1) for base_name in base_names}
best_cfg_names = {base_name : sorted(list(enumerate(cfg_names[base_name])), key=lambda x: val_scores[base_name][x[0]])[-1][1] for base_name in base_names}

print(best_cfg_names)

with open('./configs/configs_random.yaml', 'r') as f:
    rand_cfgs = yaml.load(f, Loader=yaml.SafeLoader)

best_cfgs = {base_name + 'Best' : rand_cfgs[best_cfg_names[base_name]] for base_name in base_names}

if os.path.exists('./configs/configs_best.yaml'):
    os.remove('./configs/configs_best.yaml')
with open('./configs/configs_best.yaml', 'w') as f:
    yaml.dump(best_cfgs, f)