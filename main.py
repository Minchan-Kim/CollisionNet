import tensorflow as tf
import argparse
import network as nn
import dataset as ds
import wandb
import datetime
import os
import time
import sys


class MetricsLog(tf.keras.callbacks.Callback):
    def __init__(self, wandb, **kwargs):
        super(MetricsLog, self).__init__()
        self.wandb = wandb
        self.max_val_acc = 0.0
        self.min_val_loss = 0.0
        self.use_wandb = kwargs.get('use_wandb', False)
        self.early_stopping = kwargs.get('early_stopping', '')
        self.name = kwargs.get('run_name')
        self.filename = '/home/dyros/mc_ws/CollisionNet/model/' + run_name + '.h5'

    def on_epoch_end(self, epoch, logs = None):
        if use_wandb == True:
            wandb_dict = dict()
            wandb_dict['Training Accuracy'] = logs.get('acc')
            wandb_dict['Validation Accuracy'] = logs.get('val_acc')
            wandb_dict['Training Cost'] =  logs.get('loss')
            wandb_dict['Validation Cost'] = logs.get('val_loss')
            self.wandb.log(wandb_dict)
        if (early_stopping == 'acc') and (run_name != ''):
            if (epoch == 0) or (logs.get('val_acc') > self.max_val_acc):
                model.save_weights(self.filename)
                self.min_val_loss = logs.get('val_acc')
        elif (early_stopping == 'loss') and (run_name != ''):
            if (epoch == 0) or (logs.get('val_loss') < self.min_val_loss):
                model.save_weights(self.filename)
                self.min_val_loss = logs.get('val_loss')


parser = argparse.ArgumentParser()
parser.add_argument('-profile', action = 'store_true')
parser.add_argument('-use_cpu', action = 'store_false')
parser.add_argument('-use_wandb', action = 'store_true')
parser.add_argument('--name', type = str, default = '')
parser.add_argument('--num_data', type = int, default = 49)
parser.add_argument('--time_window', type = int, default = 32)
parser.add_argument('--training_epoch', type = int, default = 100)
parser.add_argument('--buffer_size', type = int, default = 20000)
parser.add_argument('--batch_size', type = int, default = 100)
parser.add_argument('--minibatch_size', type = int, default = 0)
parser.add_argument('--early_stopping', type = str, default = '')
parser.add_argument('-schedule', action = 'store_true')
parser.add_argument('--learning_rate', type = float, default = 0.001)
parser.add_argument('--beta_1', type = float, default = 0.9)
parser.add_argument('--beta_2', type = float, default = 0.999)
parser.add_argument('--epsilon', type = float, default = 1e-7)
parser.add_argument('--cycle_length', type = int, default = 8)
parser.add_argument('--dtype', type = str, default = '')
parser.add_argument('--tool', type = str, default = '')
args = parser.parse_args()

use_gpu = args.use_cpu
use_wandb = args.use_wandb
run_name = args.name
num_data = args.num_data
time_window = args.time_window
training_epochs = args.training_epoch
buffer_size = args.buffer_size
batch_size = args.batch_size
minibatch_size = args.minibatch_size
early_stopping = args.early_stopping
use_schedule = args.schedule
learning_rate = args.learning_rate
beta_1 = args.beta_1
beta_2 = args.beta_2
epsilon = args.epsilon
cycle_length = args.cycle_length
dtype = args.dtype
tool = args.tool

if use_wandb == True:
    wandb.init(project = "collisionnet", name = run_name)

#VALIDATION_DATA = 16965
#VALIDATION_DATA = 5598

if not tf.__version__.startswith('2'):
    print('Use TensorFlow 2.X!')
    sys.exit(0)

if use_gpu is True:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if use_wandb == True:
    wandb.config.num_data = num_data
    wandb.config.time_window = time_window
    wandb.config.epoch = training_epochs
    wandb.config.buffer_size = buffer_size
    wandb.config.batch_size = batch_size
    wandb.config.early_stopping = early_stopping
    wandb.config.use_schedule = use_schedule
    wandb.config.learning_rate = learning_rate
    wandb.config.beta_1 = beta_1
    wandb.config.beta_2 = beta_2
    wandb.config.epsilon = epsilon
    wandb.config.cycle_length = cycle_length
    if tool == '':
        wandb.config.tool = 'all'
    else:
        wandb.config.tool = tool

if use_schedule:
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, 1000, 0.99)

if minibatch_size > 0:
    model = nn.CollisionNet(
        num_data, time_window, 
        bias_initializer = 'glorot_uniform',
        learning_rate = learning_rate, 
        beta_1 = beta_1, 
        beta_2 = beta_2, 
        epsilon = epsilon, 
        batch_size = batch_size, 
        minibatch_size = minibatch_size
    )
    batch_size = minibatch_size
else:
    model = nn.CollisionNet(
        num_data, time_window, 
        bias_initializer = 'glorot_uniform',
        learning_rate = learning_rate,
        beta_1 = beta_1,
        beta_2 = beta_2,
        epsilon = epsilon
    )

# Training data
if (dtype != '') and (tool != ''):
    pattern = '*' + dtype + '_' + tool + '.tfrecord'
elif dtype != '':
    pattern = '*' + dtype + '*.tfrecord'
elif tool != '':
    pattern = '*' + tool + '.tfrecord'
else:
    pattern = '*.tfrecord'
dataset = ds.Dataset(
    ('/home/dyros/mc_ws/CollisionNet/data/' + str(time_window) + '/training'), 
    num_data, time_window, buffer_size, batch_size, pattern = pattern
)

# Validation data 
if tool != '':
    pattern = '*' + tool + '.tfrecord'
else:
    pattern = '*.tfrecord'
validation_dataset = ds.Dataset(
    ('/home/dyros/mc_ws/CollisionNet/data/' + str(time_window) + '/validation'), 
    num_data, time_window, 0, batch_size, pattern = pattern, num_parallel_calls = 3, processed = False, drop_remainder = False
)

callbacks = [MetricsLog(wandb, use_wandb = use_wandb, run_name = run_name, early_stopping = early_stopping)]

start_time = time.time()
model.fit(x = dataset, epochs = training_epochs, callbacks = callbacks, validation_data = validation_dataset)
elapsed_time = time.time() - start_time

if (early_stopping == False) and (run_name != ''):
    model.save_weights('/home/dyros/mc_ws/CollisionNet/model/' + run_name + '.h5')
if use_wandb == True:
    model.save(os.path.join(wandb.run.dir, (run_name + '.h5')))
    wandb.config.elapsed_time = elapsed_time