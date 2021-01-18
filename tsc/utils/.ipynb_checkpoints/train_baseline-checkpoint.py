import sys
import os
import time
import numpy as np
import tensorflow as tf

import pickle

import keras
from keras import backend as K

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, classification_report

from utils import load_dataset

def train_baseline(dataset_name, model, normalize_data=True, batch_size=16, epochs=2000, val_split=None, test_split=None,
                   opt=None, monitor='loss', save_dir=None, experiment_idx=None):
    
    save_dir = save_dir or './results_baselines/{}/'.format(model.name.split('_')[0])
    experiment_name = '{}_{}'.format(dataset_name, '_'.join(model.name.split('_')[1:])) if len(model.name.split('_')) > 1 else '{}'.format(dataset_name)
    if experiment_idx is not None:
        experiment_name += '_{}'.format(experiment_idx)
    
    # load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_name, val_split=val_split, test_split=test_split, normalize_data=normalize_data)
    
    num_train, len_streams, num_features = X_train.shape
    num_val = X_val.shape[0] if val_split is not None else None
    num_test = X_test.shape[0]
    num_classes = np.unique(y_train).size

    y_train_1hot = keras.utils.to_categorical(y_train)
    y_val_1hot = keras.utils.to_categorical(y_val) if val_split is not None else None
    y_test_1hot = keras.utils.to_categorical(y_test)
    
    opt = opt or keras.optimizers.Adam()
    batch_size = int(min(num_train/10, batch_size))
        
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    # set callbacks from baseline
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
    weights_file = os.path.join(save_dir, experiment_name + '.hdf5')
    mc = keras.callbacks.ModelCheckpoint(weights_file, monitor=monitor, save_best_only=True, save_weights_only=True, verbose=1)
    
    class TimeHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.times = []

        def on_epoch_begin(self, epoch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, epoch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)
        
    fit_time = TimeHistory()
    
    callback_list = [reduce_lr, mc, fit_time]

    val_data = (X_val, y_val_1hot) if val_split is not None else None # (X_test, y_test_1hot)

    # fit model
    history = model.fit(X_train, y_train_1hot, batch_size=batch_size, epochs=epochs, callbacks=callback_list, verbose=1, validation_data=val_data)
    
    # restore best weights
    model.load_weights(weights_file)

    # evaluate model performance
    history = history.history
    history['time'] = fit_time.times
    history['results'] = {}
    write_to_txt = ''
    if val_split is not None:
        ##  evaluate on validation set
        y_val_pred = np.argmax(model.predict(X_val), axis=1)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_report = classification_report(y_val, y_val_pred)
        
        history['results']['val_acc'] = val_acc
        history['results']['val_report'] = val_report
        
        write_to_txt += 'Val. acc.: {:.3f}\n'.format(val_acc)
        write_to_txt += 'Val. report:\n{}\n'.format(val_report)
        
        print('Val. acc.: {:.3f}'.format(val_acc))
        print('Val. report:\n{}\n'.format(val_report))

    ## evaluate on test set
    y_test_pred = np.argmax(model.predict(X_test), axis=1)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_report = classification_report(y_test, y_test_pred)
    
    history['results']['test_acc'] = test_acc
    history['results']['test_report'] = test_report
        
    write_to_txt += 'Test acc.: {:.3f}\n'.format(test_acc)
    write_to_txt += 'Test report:\n{}\n'.format(test_report)
        
    print('Test acc.: {:.3f}'.format(test_acc))
    print('Test report:\n{}\n'.format(test_report))

    pkl_file = os.path.join(save_dir, experiment_name + '.pkl')
    with open(pkl_file, 'wb') as f:
        pickle.dump(history, f)
        
    txt_file = os.path.join(save_dir, experiment_name + '.txt')
    with open(txt_file, 'w') as f:
        f.write(write_to_txt)
    
    return