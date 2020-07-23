import tensorflow as tf
import argparse
import network as nn
import dataset as ds
import wandb
import datetime
import os
import time
import sys


parser = argparse.ArgumentParser()
parser.add_argument('-profile', action = 'store_true')
parser.add_argument('-use_wandb', action = 'store_true')
parser.add_argument('-use_cpu', action = 'store_false')
parser.add_argument('--name', type = str, default = '')
parser.add_argument('--learning_rate', type = float, default = 0.001)
parser.add_argument('--training_epoch', type = int, default = 100)
parser.add_argument('--num_data', type = int, default = 49)
parser.add_argument('--time_window', type = int, default = 32)
parser.add_argument('--buffer_size', type = int, default = 20000)
parser.add_argument('--batch_size', type = int, default = 50)
parser.add_argument('--cycle_length', type = int, default = 8)
parser.add_argument('--dtype', type = str, default = '')
parser.add_argument('--tool', type = str, default = '')
args = parser.parse_args()

use_gpu = args.use_cpu
wandb_use = args.use_wandb
run_name = args.name

if wandb_use == True:
    wandb.init(project = "collisionnet", name = run_name)

learning_rate = args.learning_rate
training_epochs = args.training_epoch
batch_size = args.batch_size
buffer_size = args.buffer_size
num_data = args.num_data
time_window = args.time_window
cycle_length = args.cycle_length
dtype = args.dtype
tool = args.tool

#VALIDATION_DATA = 16966
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

model = nn.CollisionNet(num_data, time_window, learning_rate)

if wandb_use == True:
    wandb.config.epoch = training_epochs
    wandb.config.learning_rate = learning_rate
    wandb.config.buffer_size = buffer_size
    wandb.config.batch_size = batch_size
    wandb.config.num_data = num_data
    wandb.config.time_window = time_window
    wandb.config.cycle_length = cycle_length

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
    ('/home/dyros/mc_ws/CollisionNet/data/'+ str(time_window) + '/validation_as_training'), 
    num_data, time_window, buffer_size, batch_size, pattern = pattern, cycle_length = cycle_length
)

# Validation data 
if tool != '':
    pattern = '*' + tool + '.tfrecord'
else:
    pattern = '*.tfrecord'
validation_dataset = ds.Dataset(
    ('/home/dyros/mc_ws/CollisionNet/data/' + str(time_window) + '/test'), 
    num_data, time_window, 0, batch_size, pattern = pattern, num_parallel_calls = 3, processed = False, drop_remainder = False
)

def on_epoch_end(epoch, logs = None):
    if wandb_use == True:
        wandb_dict = dict()
        wandb_dict['Training Accuracy'] = logs.get('acc')
        wandb_dict['Validation Accuracy'] = logs.get('val_acc')
        wandb_dict['Training Cost'] =  logs.get('loss')
        wandb_dict['Validation Cost'] = logs.get('val_loss')
        wandb.log(wandb_dict)

callbacks = [on_epoch_end]

start_time = time.time()
model.fit(x = dataset, epochs = training_epochs, callbacks = callbacks, validation_data = validation_dataset)
elapsed_time = time.time() - start_time

if run_name != '':
    model.save('/home/dyros/mc_ws/CollisionNet/model/' + run_name + '.h5')
if wandb_use == True:
    model.save(os.path.join(wandb.run.dir, (run_name + '.h5')))
    wandb.config.elapsed_time = elapsed_time

print("Elapsed time: {:.2f}".format(elapsed_time))
print('Learning finished!')