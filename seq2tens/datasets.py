import os
import yaml
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

from tensorflow.keras.preprocessing import sequence

from sklearn.preprocessing import StandardScaler, LabelEncoder

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_available_datasets(yaml_file='datasets.yaml', return_data_dict=False):
    """
    Returns a list of the available datasets for download contained within the local yaml file.

    Args:
        yaml_file (str, optional): A local database file which contains information about the available datasets. Defaults to 'datasets.yaml'.
        return_data_dict (bool, optional): Whether to return the dictionary loaded from the database file. Defaults to False.
    """
    
    yaml_fp = os.path.join(os.path.dirname(__file__), yaml_file)

    with open(yaml_fp, 'r') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    
    datasets = list(data_dict.keys())
    if return_data_dict:
        return datasets, data_dict
    else:
        return datasets

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def preprocess_sequence_dataset(X, normalize_data=True, pad_sequences=False, multiple_splits=True):
    """
    Jointly preprocesses all input splits of a sequence dataset, i.e. where all inputs are sequences (e.g. time series classification).

    Args:
        X (np.ndarray or list): if multiple_splits then X contains all input splits from the dataset i.e. X = [X_train, X_val, X_test] or X = [X_train, X_test],
                                while if multiple_splits==False, then X should be a single split from a dataset, e.g. X = X_train or X = X_test
        normalize_data (bool, optional): Whether to normalize the features. Defaults to True.
        pad_sequences (bool, optional): Whether to pad the sequence to equal length. Defaults to False.
        multiple_splits (bool, optional): Whether X contains multiple splits from a dataset or just a single one. Defaults to True.

    Returns:
        X1, X2, ... (np.ndarrays): preprocessed input splits (unpacked)
    """
    
    if not multiple_splits:
        X = [X]
        Y = [Y] if Y is not None else None
    
    num_train = len(X[0])
    if normalize_data:
        scaler = StandardScaler()
        if num_train <= 100:
            X = [[scaler.fit_transform(seq) for seq in x] for x in X]
        else:
            scaler.fit(np.concatenate(X[0], axis=0))
            X = [[scaler.transform(seq) for seq in x] for x in X]

    if pad_sequences:
        maxlen = max([np.max([seq.shape[0] for seq in x]) for x in X])
        X = [sequence.pad_sequences(x, maxlen=maxlen, dtype=x[0].dtype) for x in X]
        
    return X
            
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def preprocess_labels(Y, multiple_splits=True):
    """
    Preprocesses the labels in a TSC dataset.

    Args:
        Y (np.ndarray or list): if multiple_splits then Y contains all label splits from the dataset i.e. Y = [y_train, y_val, y_test] or Y = [y_train, y_test],
                                while if multiple_splits==False, then X should be a single split from a dataset, e.g. y = y_train or y = y_test
        multiple_splits (bool, optional): Whether Y contains multiple splits from a dataset or just a single one. Defaults to True.

    Returns:
        Y1, Y2, ... (np.ndarrays): preprocessed label splits (unpacked)
    """
    
    if not multiple_splits:
        Y = [Y]
    
    label_enc = LabelEncoder()
    label_enc.fit(Y[0])
    Y = [label_enc.transform(y) for y in Y]
    
    return Y
        

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_dataset(ds_name, normalize_data=True, pad_sequences=False, cache_dir=None, yaml_file='datasets.yaml'):
    """
    Loads a given dataset from a download link contained within a local database file. If it was downloaded before, the cached version is used.

    Args:
        ds_name (str): Name of the dataset. Available datasets can be queried by the get_available_datasets function.
        normalize_data (bool, optional): Whether to normalize the dataset. Defaults to True.
        pad_sequences (bool, optional): Whether to pad the dataset by prepending 0's until the length of the longest sequence is reached. Defaults to False.
        cache_dir (str, optional): A path to directory where the dataset will be cached, if None then the default is '~/.keras/datasets/'. Defaults to None.
        yaml_file (str, optional): A local database file which contains information about the available datasets. Defaults to 'datasets.yaml'.
    
    Raises:
        ValueError: Raised if ds_name is not found within the yaml_file.
        ValueError: Rised if cache_dir is not a valid directory.

    Returns:
        X_train (np.ndarray or list): if pad_sequences==True, a numpy array of shape (num_train, len_sequences, num_features)
                                      else a list of (num_train,) arrays, each of shape (len_sequence, num_features), where len_sequence can vary from instance to instance
        y_train (np.ndarray): a numpy array of (num_train,) training labels
        X_test (np.ndarray or list): if pad_sequences==True, a numpy array of shape (num_test, len_sequences, num_features)
                                     else a list of (num_test,) arrays, each of shape (len_sequence, num_features), where len_sequence can vary from instance to instance
        y_test (np.ndarray): a numpy array of (num_test,) training labels
    """   
    
    datasets, data_dict = get_available_datasets(return_data_dict=True)
    
    if ds_name not in datasets:
        raise ValueError(f'ValueError | load_dataset: dataset \'{ds_name}\' not found.')
    
    if cache_dir is not None and not os.path.isdir(cache_dir):
        raise ValueError(f'ValueError | load_dataset: cache_dir \'{cache_dir}\' is not a valid directory.')

    fp = get_file(ds_name, data_dict[ds_name]['link'], file_hash=data_dict[ds_name]['hash'], cache_dir=None)
    with np.load(fp, allow_pickle=True, encoding='bytes') as f:
        X_train, y_train = f['X_train'], f['y_train']
        X_test, y_test = f['X_test'], f['y_test']
        
    print(f'Information | load_dataset: Dataset \'{ds_name}\' successfully loaded.')
    
    X_train, y_train, X_test, y_test = np.squeeze(X_train), np.squeeze(y_train), np.squeeze(X_test), np.squeeze(y_test)
    
    X_train, X_test = preprocess_sequence_dataset((X_train, X_test), normalize_data=True, pad_sequences=True, multiple_splits=True)
    y_train, y_test = preprocess_labels((y_train, y_test), multiple_splits=True)
    
    return X_train, y_train, X_test, y_test

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------