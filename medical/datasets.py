import sys
import numpy as np

import tensorflow_datasets as tfds
import medical_ts_datasets

from tensorflow.keras import preprocessing

sys.path.append('..')

from seq2tens.datasets import preprocess_sequence_dataset, preprocess_labels

def load_medical_dataset(dataset_name, add_time=False, add_static=False, add_mask=False, add_tmask=False, normalize_data=True, pad_sequences=True, **kwargs):
    
    data = tfds.as_numpy(tfds.load(dataset_name))
    
    data_train = list(data['train'])
    data_val = list(data['validation'])
    data_test = list(data['test'])
    
    feature_means = compute_feature_means(data_train)
    
    X_train, y_train = prepare_data_features(data_train, add_time=add_time, add_static=add_static, add_mask=add_mask, add_tmask=add_tmask, default_val=feature_means)
    X_val, y_val = prepare_data_features(data_val, add_time=add_time, add_static=add_static, add_mask=add_mask, add_tmask=add_tmask, default_val=feature_means)
    X_test, y_test = prepare_data_features(data_test, add_time=add_time, add_static=add_static, add_mask=add_mask, add_tmask=add_tmask, default_val=feature_means)
    
    X_train, X_val, X_test = preprocess_sequence_dataset((X_train, X_val, X_test), normalize_data=normalize_data, pad_sequences=pad_sequences)
    y_train, y_val, y_test = preprocess_labels((y_train, y_val, y_test), multiple_splits=True)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def compute_feature_means(data_train):
    means = np.nanmean(np.concatenate([x['combined'][2] for x in data_train], axis=0), axis=0)
    return means

def prepare_data_features(data, add_time=False, add_static=False, add_mask=False, add_tmask=False, default_val=None):
    mask = [x['combined'][3] for x in data]
    time = [x['combined'][1] for x in data]
    ts = [x['combined'][2] for x in data]
    static = [x['combined'][0] for x in data]
    y = [x['target'] for x in data]

    if default_val is None:
        default_val = np.zeros(ts[0][0].shape)
    def impute(x, m):
        x[0, np.logical_not(m[0])] = default_val[np.logical_not(m[0])]
        for i in range(1, x.shape[0]):
            x[i, np.logical_not(m[i])] = x[i-1, np.logical_not(m[i])]
        return x
    X = [np.float64(impute(x, mask[i])) for i, x in enumerate(ts)]
    if add_time:
        X = [np.concatenate((time[i][:, None], x), axis=1) for i, x in enumerate(X)]
    if add_static:
        X = [np.concatenate((np.tile(static[i][None, :], [x.shape[0], 1]), x), axis=1) for i, x in enumerate(X)]
    if add_mask:
        X = [np.concatenate((mask[i], x), axis=1) for i, x in enumerate(X)]
    if add_tmask:
        for i in range(len(mask)):
            mask[i][0, :] = 1
        tmask = [np.stack([time[i][np.bool_(m[:, d])][np.cumsum(m[:, d])-1] for d in range(m.shape[1])], axis=1) for i, m in enumerate(mask)]
        X = [np.concatenate((tmask[i], x), axis=1) for i, x in enumerate(X)]
    return X, y