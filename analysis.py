import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataset as ds
from pathlib import Path
import argparse


class FalsePositives(tf.keras.metrics.Metric):
    def __init__(self, thresholds = 0.5, name = 'false_positives', **kwargs):
        super(FalsePositives, self).__init__(name = name, **kwargs)
        self.false_positives = self.add_weight(name = 'fp', initializer = 'zeros')
        self.thresholds = thresholds

    def update_state(self, y_true, y_pred, sample_weight = None):
        y_true = tf.cast(y_true[:, 0], tf.bool)
        y_pred = tf.math.greater_equal(y_pred[:, 0], self.thresholds)
        values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_weights(sample_weight, values)
            values = tf.math.multiply(values, sample_weight)
        self.false_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.false_positives


class FalseNegatives(tf.keras.metrics.Metric):
    def __init__(self, thresholds = 0.5, name = 'false_negatives', **kwargs):
        super(FalseNegatives, self).__init__(name = name, **kwargs)
        self.false_negatives = self.add_weight(name = 'fn', initializer = 'zeros')
        self.thresholds = thresholds

    def update_state(self, y_true, y_pred, sample_weight = None):
        y_true = tf.cast(y_true[:, 0], tf.bool)
        y_pred = tf.math.greater_equal(y_pred[:, 0], self.thresholds)
        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_weights(sample_weight, values)
            values = tf.math.multiply(values, sample_weight)
        self.false_negatives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.false_negatives


parser = argparse.ArgumentParser()
parser.add_argument('name', nargs = '+')
parser.add_argument('--dtype', type = str, default = '')
parser.add_argument('--tool', type = str, default = '')
args = parser.parse_args()

name = args.name[0]
dtype = args.dtype
tool = args.tool

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

model = tf.keras.models.load_model(('/home/dyros/mc_ws/CollisionNet/model/' + name + '.h5'), compile = False)
model.compile(
    loss = 'categorical_crossentropy', 
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(), 
        FalsePositives(), 
        FalseNegatives(),
        ])

if (dtype != '') and (tool != ''):
    pattern = '*' + dtype + '_' + tool + '.tfrecord'
elif dtype != '':
    pattern = '*' + dtype + '*.tfrecord'
elif tool != '':
    pattern = '*' + tool + '.tfrecord'
else:
    pattern = '*.tfrecord'

directory = '/home/dyros/mc_ws/CollisionNet/data/validation'
Paths = Path(directory).glob(pattern)
paths = [str(path) for path in Paths]
dataset = ds.OrderedDataset(paths, 49, 32, 100)
result = model.evaluate(dataset, verbose = 0)
predictions = model.predict(dataset)

false_positives = int(result[2])
false_negatives = int(result[3])
print('Accuracy: {:.2}'.format(result[1]))
print('False positives: {}'.format(false_positives))
print('False negatives: {}'.format(false_negatives))

paths = [(path[:-8] + 'csv') for path in paths]
dataframes = [pd.read_csv(path, header = None, skiprows = 31) for path in paths]
dataframe = pd.concat(dataframes, ignore_index = True)
labels = dataframe.to_numpy()

length = (labels.shape)[0]
y = labels[:, 0]
positives = np.sum(y)
print("False negative rate: {:.3}".format(false_negatives / float(positives)))

JTS = labels[:, 2]
DOB = labels[:, 3]

start = 0
for i in range(length):
    if y[i] == 1:
        start = i
        break
start = max([(start - 100), 0])
stop = min([(start + 2000), length])
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
plt.savefig('/home/dyros/mc_ws/CollisionNet/data_0_00kg/analysis/' + name '.png')
plt.clf()