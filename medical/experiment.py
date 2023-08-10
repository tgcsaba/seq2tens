import sys
import os
import copy

import yaml

import numpy as np
import tensorflow as tf

from tensorflow.data.experimental import AUTOTUNE

GPUS = tf.config.list_physical_devices('GPU')
if len(GPUS) > 0:
    for GPU in GPUS:
        tf.config.experimental.set_memory_growth(GPU, True)

from tensorflow import keras

sys.path.append('..')
import seq2tens

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from sacred import Experiment, Ingredient

from datasets import load_medical_dataset


# define dataset ingredient
data_ingredient = Ingredient('dataset')
# load model configs
with open('./configs/datasets.yaml', 'r') as f:
    datasets = yaml.load(f, Loader=yaml.SafeLoader)

for dataset_name in datasets:
    data_ingredient.add_named_config(dataset_name, datasets[dataset_name])
data_ingredient.add_config(list(datasets.values())[0]) # by default, use first dataset from the list


# define experiment
ex = Experiment('medical', ingredients=[data_ingredient])

# load model configs
with open('./configs/configs_default.yaml', 'r') as f:
    configs = yaml.load(f, Loader=yaml.SafeLoader)
# add default configs as named configs
for config_name in configs:
    configs[config_name]['config_name'] = config_name
    ex.add_named_config(config_name, configs[config_name])

if os.path.exists('./configs/configs_random.yaml'):
    with open('./configs/configs_random.yaml', 'r') as f:
        configs = yaml.load(f, Loader=yaml.SafeLoader)
    # add random configs as named configs
    for config_name in configs:
        configs[config_name]['config_name'] = config_name
        ex.add_named_config(config_name, configs[config_name])
        
if os.path.exists('./configs/configs_best.yaml'):
    with open('./configs/configs_best.yaml', 'r') as f:
        configs = yaml.load(f, Loader=yaml.SafeLoader)
    # add best configs as named configs
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
def run(_run, dataset, model_name, model_hparams, train_hparams):
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_medical_dataset(**dataset)
    
    num_classes = np.unique(y_train).size
    if num_classes != 2:
        raise NotImplementedError('The medical experiment does not support multi-class datasets.')
    
    model = seq2tens.models.get_classifier_by_name(model_name)(num_classes, mode='binary', **copy.deepcopy(train_hparams), **copy.deepcopy(model_hparams))

    
    batch_size = train_hparams['batch_size'] if 'batch_size' in train_hparams else 4
    num_ex_per_class = np.bincount(y_train)
    steps_per_epoch = np.min([int(np.ceil(n / (float(batch_size) / num_classes))) for n in num_ex_per_class])
    val_steps = int(np.ceil(len(X_val) / float(batch_size)))
    
    ds_train = seq2tens.utils.make_balanced_generator(X_train, y_train).batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).repeat().batch(batch_size).prefetch(AUTOTUNE)

    logger_cb = seq2tens.utils.SacredLogger(_run)
    val_cb = seq2tens.utils.ValidationMetrics({'auroc' : roc_auc_score, 'auprc' : average_precision_score}, X_val, y_val, batch_size=batch_size, verbose=True)

    history = model.fit_generator(ds_train, steps_per_epoch=steps_per_epoch, callbacks=[val_cb, logger_cb], validation_data=ds_val, validation_steps=val_steps)
      
#     num_trainable =  count_params(model.trainable_weights)
#     num_non_trainable = count_params(model.non_trainable_weights)
    
#     _run.log_scalar('trainable_params', num_trainable)
#     _run.log_scalar('non_trainable_params', num_non_trainable)
#     _run.log_scalar('total_params', num_trainable + num_non_trainable)
    
    train_loss, _ =  model.evaluate(X_train, y_train, batch_size=batch_size)
    y_pred_train  = model.predict(X_train, batch_size=batch_size)
    train_acc = accuracy_score(y_train, y_pred_train > 0.5)
    train_auroc = roc_auc_score(y_train, y_pred_train)
    train_auprc = average_precision_score(y_train, y_pred_train)
    _run.log_scalar('training.final_loss', train_loss)
    _run.log_scalar('training.final_accuracy', train_acc)
    _run.log_scalar('training.final_auroc', train_auroc)
    _run.log_scalar('training.final_auprc', train_auprc)
    
    val_loss, _ =  model.evaluate(X_val, y_val, batch_size=batch_size)
    y_pred_val  = model.predict(X_val, batch_size=batch_size)
    val_acc = accuracy_score(y_val, y_pred_val > 0.5)
    val_auroc = roc_auc_score(y_val, y_pred_val)
    val_auprc = average_precision_score(y_val, y_pred_val)
    _run.log_scalar('validation.final_loss', val_loss)
    _run.log_scalar('validation.final_accuracy', val_acc)
    _run.log_scalar('validation.final_auroc', val_auroc)
    _run.log_scalar('validation.final_auprc', val_auprc)
    
    test_loss, _ = model.evaluate(X_test, y_test, batch_size=batch_size)
    y_pred_test  = model.predict(X_test, batch_size=batch_size)
    test_acc = accuracy_score(y_test, y_pred_test > 0.5)
    test_auroc = roc_auc_score(y_test, y_pred_test)
    test_auprc = average_precision_score(y_test, y_pred_test)
    _run.log_scalar('testing.final_loss', test_loss)
    _run.log_scalar('testing.final_accuracy', test_acc)
    _run.log_scalar('testing.final_auroc', test_auroc)
    _run.log_scalar('testing.final_auprc', test_auprc)