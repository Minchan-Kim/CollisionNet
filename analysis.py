import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import network as nn
import dataset as ds
from pathlib import Path
import metrics
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('name', nargs = '+')
parser.add_argument('--time_window', type = int, default = 32)
parser.add_argument('--timesteps', type = int, default = 20000)
parser.add_argument('--dataset', type = str, default = 'test')
parser.add_argument('--dtype', type = str, default = '')
parser.add_argument('--tool', type = str, default = '')
args = parser.parse_args()

name = args.name[0]
time_window = args.time_window
timesteps = args.timesteps
dataset_type = args.dataset
dtype = args.dtype
tool = args.tool

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

#model = tf.keras.models.load_model(('/home/dyros/mc_ws/CollisionNet/model/' + name + '.h5'), compile = False)
model = nn.CollisionNet(49, time_window, compile = False)
model.load_weights('/home/dyros/mc_ws/CollisionNet/model/' + name + '.h5')
model.compile(
    loss = 'categorical_crossentropy', 
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(), 
        metrics.FalsePositives(), 
        metrics.FalseNegatives(),
        ])

log_file = open('/home/dyros/mc_ws/CollisionNet/analysis/' + name + '_on_' + dataset_type + '.txt', 'w')

if (dtype != '') and (tool != ''):
    pattern = '*' + dtype + '_' + tool + '.tfrecord'
elif dtype != '':
    pattern = '*' + dtype + '*.tfrecord'
elif tool != '':
    pattern = '*' + tool + '.tfrecord'
else:
    pattern = '*.tfrecord'

directory = '/home/dyros/mc_ws/CollisionNet/data/' + str(time_window) + '/' + dataset_type
Paths = Path(directory).glob(pattern)
paths = [str(path) for path in Paths]
if dataset_type == 'training':
    processed = True
else:
    processed = False
dataset = ds.OrderedDataset(paths, 49, time_window, 100, processed = processed)
result = model.evaluate(dataset, verbose = 0)
predictions = model.predict(dataset)

false_positives = int(result[2])
false_negatives = int(result[3])
line = 'Accuracy: {:.3}'.format(result[1])
print(line)
log_file.write(line + '\n')
line = 'False positives: {}'.format(false_positives)
print(line)
log_file.write(line + '\n')
line = 'False negatives: {}'.format(false_negatives)
print(line)
log_file.write(line + '\n')

paths = [(path[:-8] + 'csv') for path in paths]
dataframes = [pd.read_csv(path, header = None, skiprows = 31) for path in paths]
dataframe = pd.concat(dataframes, ignore_index = True)
labels = dataframe.to_numpy()

length = (labels.shape)[0]
y = labels[:, 0]
positives = np.sum(y)
line = 'False negative rate: {:.3}'.format(false_negatives / float(positives))
print(line)
log_file.write(line + '\n')

JTS = labels[:, 2]
DOB = labels[:, 3]

log_file.close()

start = 0
for i in range(length):
    if y[i] == 1:
        start = i
        break
start = max([(start - 100), 0])
stop = min([(start + timesteps), length])
t = range(start, stop)
plt.figure(figsize = (15, 4.8))
plt.plot(t, predictions[start:stop, 0], color = 'b', label = 'prediction')
plt.plot(t, y[start:stop], color = 'r', label = 'real')
#plt.plot(t, JTS[start:stop], color = 'k', marker = "x", label = 'jts')
#plt.plot(t, DOB[start:stop], color = 'y',marker = "x", label = 'dob')
plt.xlabel('time')
plt.ylabel('Collision Probability')
plt.legend()
plt.xlim(start, (stop - 1))
plt.ylim(0, 1.1)
plt.savefig('/home/dyros/mc_ws/CollisionNet/analysis/' + name + '_on_' + dataset_type + '.png')
plt.clf()