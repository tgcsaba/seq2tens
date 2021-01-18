from utils import train_model
from models import init_ls2t_model

import sys
import os
import json

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from mixture import SWATS

if len(sys.argv) > 1:
    GPU_ID = str(sys.argv[1])
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

with open('./datasets.json', 'r') as f:
    datasets = json.load(f)
    
    
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

                 
num_experiments = 4
batch_size = 4


preprocess_size = 64
ls2t_size = 64
ls2t_order = 2
ls2t_depth = 3
recursive_tensors = True
preprocess_time = True
monitor_test = False


for preprocess in ['', 'conv']:
    for i in range(num_experiments):
        model_name = init_ls2t_model(preprocess_size, ls2t_size, ls2t_order, ls2t_depth, preprocess=preprocess, preprocess_time=preprocess_time, recursive_tensors=recursive_tensors, name_only=True)
        model_name, model_args = model_name.split('_')[0], '_'.join(model_name.split('_')[1:])

        save_dir = './results/{}/'.format(model_name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        for dataset in datasets:

            results_file = os.path.join(save_dir, '{}_{}_{}.txt'.format(dataset, model_args, i))

            if os.path.exists(results_file):
                print('{} already exists, continuing...'.format(results_file))
                continue

            with open(results_file, 'w'):
                pass

            num_train = datasets[dataset]['n_train']
            len_sequences = datasets[dataset]['l_max']
            num_features = datasets[dataset]['n_features']
            input_shape = (len_sequences, num_features, )

            num_classes = datasets[dataset]['n_classes']

#             with tf.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            tf_opt = SWATS(1e-3, clipvalue=1.)
            model = init_ls2t_model(preprocess_size, ls2t_size, ls2t_order, ls2t_depth, preprocess=preprocess, preprocess_time=preprocess_time,
                                     recursive_tensors=recursive_tensors, input_shape=input_shape, num_classes=num_classes)
            train_model(dataset, model, normalize_data=True, batch_size=batch_size, balance_loss=True, save_dir=save_dir, experiment_idx=i, monitor_test=monitor_test, opt=tf_opt)
