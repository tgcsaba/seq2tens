import sys
import os
sys.path.append('../..')
import numpy as np

from scipy.io import loadmat

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

from scipy.interpolate import interp1d

from keras.preprocessing import sequence

def load_dataset(dataset_name, normalize_data=False, val_split=None, test_split=None):
    """ Loads a given MTSC dataset
    
    """
    
    # if test_split is not None it will instead return test_split % of the training data for testing

    data_path = './datasets/{}.mat'.format(dataset_name)
   
    if not os.path.exists(data_path):
        raise ValueError('Please download the attached datasets and extract to the /benchmarks/datasets/ directory...')
        
    data = loadmat(data_path)
    
    X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
    
    X_train, y_train, X_test, y_test = np.squeeze(X_train), np.squeeze(y_train), np.squeeze(X_test), np.squeeze(y_test)
    
    num_train = len(X_train)
    num_test = len(X_test)
    
    num_features = X_train[0].shape[1]
    num_classes = np.unique(np.int32(y_train)).size
    
    if val_split is not None:
        if val_split < 1. and np.ceil(val_split * num_train) < 2 * num_classes:
            val_split = 2 * num_classes
        elif val_split > 1. and val_split < 2 * num_classes:
            val_split = 2 * num_classes
    
    if test_split is not None:
        if test_split < 1. and np.ceil(test_split * num_train) < 2 * num_classes:
            test_split = 2 * num_classes
        elif test_split > 1. and test_split < 2 * num_classes:
            test_split = 2 * num_classes
    
    if val_split is not None and test_split is not None:
        if val_split < 1. and test_split > 1:
            val_split = int(np.ceil(num_train * val_split))
        elif val_split > 1 and test_split < 1.:
            test_split = int(np.ceil(num_train * test_split))
                
    split_from_train = val_split + test_split if val_split is not None and test_split is not None else val_split or test_split 

    if split_from_train is not None:

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=split_from_train, shuffle=True, stratify=y_train)
        
        if val_split is not None and test_split is not None:
            X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=float(test_split)/split_from_train, shuffle=True, stratify=y_val)
            num_val = len(X_val)
            num_test = len(X_test)
        elif val_split is not None:
            num_val = len(X_val)
        else:
            X_test, y_test = X_val, y_val
            X_val, y_val = None, None
            num_test = len(X_test)
            num_val = 0
        num_train = len(X_train)
    else:
        X_val, y_val = None, None
        num_val = 0

    if normalize_data:
        scaler = StandardScaler()
        if dataset_name == 'CMUsubject16' or dataset_name == 'KickvsPunch' or dataset_name == 'ECG':
            X_train = [scaler.fit_transform(x) for x in X_train]
            X_val = [scaler.fit_transform(x) for x in X_val] if X_val is not None else None
            X_test = [scaler.fit_transform(x) for x in X_test]
        else:
            scaler.fit(np.concatenate(X_train, axis=0))
            X_train = [scaler.transform(x) for x in X_train]
            X_val = [scaler.transform(x) for x in X_val] if X_val is not None else None
            X_test = [scaler.transform(x) for x in X_test]
    
    if X_val is None:
        maxlen = max(np.max([x.shape[0] for x in X_train]), np.max([x.shape[0] for x in X_test]))
        X_train = sequence.pad_sequences(X_train, maxlen=maxlen, dtype=X_train[0].dtype)
        X_test = sequence.pad_sequences(X_test, maxlen=maxlen, dtype=X_train[0].dtype)
    else:
        maxlen = max(np.max([x.shape[0] for x in X_train]), np.max([x.shape[0] for x in X_val]), np.max([x.shape[0] for x in X_test]))
        X_train = sequence.pad_sequences(X_train, maxlen=maxlen, dtype=X_train[0].dtype)
        X_val = sequence.pad_sequences(X_val, maxlen=maxlen, dtype=X_train[0].dtype)
        X_test = sequence.pad_sequences(X_test, maxlen=maxlen, dtype=X_train[0].dtype)
    
    labels = {y : i for i, y in enumerate(np.unique(y_train))}

    y_train = np.asarray([labels[y] for y in y_train])
    y_val = np.asarray([labels[y] for y in y_val]) if y_val is not None else None
    y_test = np.asarray([labels[y] for y in y_test])
    
    return X_train, y_train, X_val, y_val, X_test, y_test
