import sys
import os

import yaml

import numpy as np
import tensorflow as tf

from tensorflow.python.keras.utils.layer_utils import count_params

GPUS = tf.config.list_physical_devices('GPU')
if len(GPUS) > 0:
    for GPU in GPUS:
        tf.config.experimental.set_memory_growth(GPU, True)

from tensorflow import keras

sys.path.append('..')
import seq2tens

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score

from sacred import Experiment, Ingredient

# define dataset ingredient
data_ingredient = Ingredient('dataset')

# load dataset specs
datasets, data_dict = seq2tens.datasets.get_available_datasets(return_data_dict=True)

for dataset_name in datasets:
    data_dict[dataset_name]['dataset_name'] = dataset_name
    data_ingredient.add_named_config(dataset_name, data_dict[dataset_name])
data_ingredient.add_config(data_dict[datasets[0]]) # by default, run first dataset from the list

# define experiment
ex = Experiment('tsc', ingredients=[data_ingredient])

# load model configs
with open('./configs.yaml', 'r') as f:
    configs = yaml.load(f, Loader=yaml.SafeLoader)

for config_name in configs:
    configs[config_name]['config_name'] = config_name
    ex.add_named_config(config_name, configs[config_name])

# default config to use if none is specified
@ex.config
def default():
    config_name = 'default'
    model_name = 'FCNLS2T'
    model_hparams = {}
    train_hparams = {}

@ex.automain
def run(_run, dataset, model_name, model_hparams, train_hparams, follow_test_metrics=False, tmp_dir='./tmp/'):
    X_train, y_train, X_test, y_test = seq2tens.datasets.load_dataset(dataset['dataset_name'], pad_sequences=True)
    
    model = seq2tens.models.get_classifier_by_name(model_name)(dataset['n_classes'], **train_hparams, **model_hparams)

    logger_cb = seq2tens.utils.SacredLogger(_run)
    
    history = model.fit(X_train, y_train, epochs=1, balance_loss=True, callbacks=[logger_cb])
    
    num_trainable =  count_params(model.trainable_weights)
    num_non_trainable = count_params(model.non_trainable_weights)
    
    _run.log_scalar('trainable_params', num_trainable)
    _run.log_scalar('non_trainable_params', num_non_trainable)
    _run.log_scalar('total_params', num_trainable + num_non_trainable)