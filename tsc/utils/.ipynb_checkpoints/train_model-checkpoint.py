import sys

sys.path.append('..')

import os
import time
import numpy as np
import tensorflow as tf

import seq2tens

import pickle

from tensorflow.keras import utils, callbacks, optimizers

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, classification_report

from utils import load_dataset

def train_model(dataset_name, model, normalize_data=True, batch_size=4, val_split=None, test_split=None, monitor_test=False,
                balance_loss=True, opt=None, monitor='loss', save_dir=None, experiment_idx=None, use_lsuv=False):
    
    save_dir = save_dir or './results/{}/'.format(model.name.split('_')[0])
    experiment_name = '{}_{}'.format(dataset_name, '_'.join(model.name.split('_')[1:]))
    if experiment_idx is not None:
        experiment_name += '_{}'.format(experiment_idx)
    
    # load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_name, val_split=val_split, test_split=test_split, normalize_data=normalize_data)
    
    num_train, len_streams, num_features = X_train.shape
    num_val = X_val.shape[0] if val_split is not None else None
    num_test = X_test.shape[0]
    num_classes = np.unique(y_train).size
    
    batch_size = max(min(batch_size, int(num_train / 10.)), 4)

    y_train_1hot = utils.to_categorical(y_train)
    y_val_1hot = utils.to_categorical(y_val) if val_split is not None else None
    y_test_1hot = utils.to_categorical(y_test)
    
    opt = opt or optimizers.Adam(1e-3, clipvalue=1.) # 
    
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    if use_lsuv:
        init_batch_size = 100
        idx_init = np.random.choice(num_train, size=(init_batch_size), replace=False)
        X_init = X_train[idx_init]
        y_init = y_train[idx_init]
        
        model = seq2tens.utils.LSUVReinitializer(model, X_init)
    
    
    sample_weight = compute_sample_weight('balanced', y_train) if balance_loss else None

    # set callbacks
    
    class TimeHistory(callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.times = []

        def on_epoch_begin(self, epoch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, epoch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)
            
    class ReduceLRBacktrack(callbacks.ReduceLROnPlateau):
        def __init__(self, best_path, *args, **kwargs):
            super(ReduceLRBacktrack, self).__init__(*args, **kwargs)
            self.best_path = best_path

        def on_epoch_end(self, epoch, logs=None):
            current = logs.get(self.monitor)
            if current is None:
                logging.warning('Reduce LR on plateau conditioned on metric `%s` '
                                'which is not available. Available metrics are: %s',
                                 self.monitor, ','.join(list(logs.keys())))
            if not self.monitor_op(current, self.best): # not new best
                if not self.in_cooldown() and float(tf.keras.backend.get_value(self.model.optimizer.lr)) > self.min_lr:
                    if self.wait+1 >= self.patience: # going to reduce lr
                        # load best model so far
                        print("Backtracking to best model before reducting LR")
                        self.model.load_weights(self.best_path)
            super().on_epoch_end(epoch, logs) # actually reduce LR
            
    class EarlyStoppingByLossVal(callbacks.Callback):
        def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
            super(EarlyStoppingByLossVal, self).__init__()
            self.monitor = monitor
            self.value = value
            self.verbose = verbose

        def on_epoch_end(self, epoch, logs={}):
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

            if current < self.value:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping THR" % epoch)
                self.model.stop_training = True
        
    es = callbacks.EarlyStopping(monitor=monitor, patience=300, restore_best_weights=True, verbose=1)
    es_loss = EarlyStoppingByLossVal(monitor=monitor, value=1e-8)
    weights_file = os.path.join(save_dir, experiment_name + '.hdf5')
    mc = callbacks.ModelCheckpoint(weights_file, monitor=monitor, save_best_only=True, save_weights_only=True, verbose=1)
#     reduce_lr = ReduceLRBacktrack(weights_file, monitor=monitor, patience=50, factor=1/np.sqrt(2.), min_lr=1e-4, verbose=1)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor=monitor, patience=50, factor=1/np.sqrt(2.), min_lr=1e-4, verbose=1)
    
    fit_time = TimeHistory()
    
    callback_list = [fit_time, mc, es, es_loss, reduce_lr]

    val_data = (X_val, y_val_1hot) if val_split is not None else (X_test, y_test_1hot) if monitor_test else None

    history = model.fit(X_train, y_train_1hot, batch_size=batch_size, epochs=10000, callbacks=callback_list, verbose=1, validation_data=val_data, shuffle=True, sample_weight=sample_weight)
    
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