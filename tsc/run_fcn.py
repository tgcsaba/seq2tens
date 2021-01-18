from utils import train_baseline
from models import init_fcn_model

import sys
import os
import json

import tensorflow as tf
import keras

from keras import backend as K

if len(sys.argv) > 1:
    GPU_ID = str(sys.argv[1])
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

with open('./datasets.json', 'r') as f:
    datasets = json.load(f)

num_experiments = 4
for i in range(num_experiments):    
    # create results folder if not exists
    save_dir = './results/FCN/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
            
    # run all datasets
    for dataset in datasets:
        
        results_file = os.path.join(save_dir, '{}_{}.txt'.format(dataset, i))

        if os.path.exists(results_file):
            print('{} already exists, continuing...'.format(results_file))
            continue

        with open(results_file, 'w'):
            pass
        
        input_shape = (datasets[dataset]['l_max'], datasets[dataset]['n_features'])
        
        num_classes = datasets[dataset]['n_classes']
        
#         with tf.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
#             K.set_session(sess)
        
        model = init_fcn_model(input_shape, num_classes)

        train_baseline(dataset, model, normalize_data=True, batch_size=16, epochs=2000, opt=keras.optimizers.Adam(), save_dir=save_dir, experiment_idx=i)