import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import network as nn
import dataset as ds
import metrics
import argparse
from pathlib import Path


class Ensemble(tf.keras.layers.Layer):
    def __init__(self, threshold):
        super(Ensemble, self).__init__()
        self.threshold = threshold

    def call(self, inputs):
        y1 = tf.gather(inputs, indices = [0], axis = 1)
        y2 = tf.gather(inputs, indices = [2], axis = 1)
        y1 = tf.math.greater(y1, self.threshold)
        y2 = tf.math.greater(y2, self.threshold)
        y = tf.logical_not(tf.math.logical_and(y1, y2))
        y = tf.squeeze(tf.cast(y, tf.int32))
        return tf.one_hot(y, 2)


def print_log(line, log_file):
    print(line)
    log_file.write(line + '\n')

def analyze_collision(threshold, predictions, y, JTS, DOB, log_file):
    collision_cnt = 0
    collision_time = 0
    detection_time_NN = []
    detection_time_JTS = []
    detection_time_DOB = []
    collision_status = False
    detection_NN = False
    detection_JTS = False
    detection_DOB = False
    collision_fail_cnt_NN = 0
    collision_fail_cnt_JTS = 0
    collision_fail_cnt_DOB = 0

    t = np.arange(0, (0.001 * len(y)), 0.001)

    for i in range(len(y)):
        if (y[i] == 1) and (collision_status == False):
            collision_cnt += 1
            collision_time = t[i]
            collision_status = True
            detection_NN = False
            detection_JTS = False
            detection_DOB = False
        
        if ((collision_status == True) and (detection_NN == False)):
            if (predictions[i, 0] == 1):
                detection_NN = True
                detection_time_NN.append(t[i] - collision_time)
        if ((collision_status == True) and (detection_JTS == False)):
            if (JTS[i] == 1):
                detection_JTS = True
                detection_time_JTS.append(t[i] - collision_time)
        if ((collision_status == True) and (detection_DOB == False)):
            if(DOB[i] == 1):
                detection_DOB = True
                detection_time_DOB.append(t[i] - collision_time)

        if (y[i] == 0) and (collision_status == True):
            collision_status = False
            if (detection_NN == False):
                detection_time_NN.append(0.0)
                collision_fail_cnt_NN += 1
            if (detection_JTS == False):
                detection_time_JTS.append(0.0)
                collision_fail_cnt_JTS += 1
            if (detection_DOB == False):
                detection_time_DOB.append(0.0)
                collision_fail_cnt_DOB += 1

    print_log('Total collision: {}'.format(collision_cnt), log_file)
    print_log('NN failure: {}'.format(collision_fail_cnt_NN), log_file)
    print_log('JTS failure: {}'.format(collision_fail_cnt_JTS), log_file)
    print_log('DOB failure: {}'.format(collision_fail_cnt_DOB), log_file)
    print_log('NN detection time: {}'.format(sum(detection_time_NN) / (collision_cnt - collision_fail_cnt_NN)), log_file)
    print_log('JTS detection time: {}'.format(sum(detection_time_JTS) / (collision_cnt - collision_fail_cnt_JTS)), log_file)
    print_log('DOB detection time: {}'.format(sum(detection_time_DOB) / (collision_cnt - collision_fail_cnt_DOB)), log_file)

def analyze_free(threshold, predictions, y, JTS, DOB, log_file):
    NN_FP_time = []
    NN_FP = 0
    JTS_FP_time = []
    JTS_FP = 0
    DOB_FP_time = []
    DOB_FP = 0
    t = np.arange(0, (0.001 * len(y)), 0.001)
    for i in range(len(y)):
        if ((predictions[i, 0] == 1) and (y[i] == 0)):
            NN_FP_time.append(t[i])
            NN_FP += 1
        if ((JTS[i] == 1) and (y[i] == 0)):
            JTS_FP_time.append(t[i])
            JTS_FP += 1
        if ((DOB[i] == 1) and (y[i] == 0)):
            DOB_FP_time.append(t[i])
            DOB_FP += 1
    
    print_log("NN FP Time: ", log_file)
    for i in range(NN_FP - 1):
        dt = abs(NN_FP_time[i + 1] - NN_FP_time[i])
        if (dt > 0.5):
            print_log(str(dt), log_file)
    print_log("JTS FP Time: ", log_file)
    for i in range(JTS_FP - 1):
        dt = abs(JTS_FP_time[i + 1] - JTS_FP_time[i])
        if (dt > 0.5):
            print_log(str(dt), log_file)
    print_log("DOB FP Time: ", log_file)
    for i in range(DOB_FP - 1):
        dt = abs(DOB_FP_time[i + 1] - DOB_FP_time[i])
        if (dt > 0.5):
            print_log(str(dt), log_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs = 2)
    parser.add_argument('-plot', action = 'store_true')
    parser.add_argument('--time_window', type = int, default = 32)
    parser.add_argument('--threshold', type = float, default = 0.99)
    parser.add_argument('--timesteps', type = int, default = 20000)

    args = parser.parse_args()
    name1 = args.name[0]
    name2 = args.name[1]
    time_window = args.time_window
    threshold = args.threshold
    timesteps = args.timesteps

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    model1 = nn.CollisionNet(49, 31, compile = False)
    model1.load_weights('/home/dyros/mc_ws/CollisionNet/model/' + name1 + '.h5')
    model2 = nn.CollisionNet(49, 31, compile = False)
    model2.load_weights('/home/dyros/mc_ws/CollisionNet/model/' + name2 + '.h5')
    inputs = tf.keras.Input(shape = (time_window, 49))
    y1 = model1(inputs)
    y2 = model2(inputs)
    outputs = tf.keras.layers.concatenate([y1, y2])
    outputs = Ensemble(threshold)(outputs)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    model.compile(
        loss = 'categorical_crossentropy', 
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(), 
            metrics.FalsePositives(threshold), 
            metrics.FalseNegatives(threshold),
        ])

    dtypes = ['collision', 'free']
    tools = ['0_00kg','2_01kg','3_33kg','5_01kg']
    directory = '/home/dyros/mc_ws/CollisionNet/data/' + str(time_window) + '/test'
    threshold = 0.99

    log_file = open('/home/dyros/mc_ws/CollisionNet/analysis/' + name1 + '_' + name2 + '_ensemble.txt', 'w')
    print_log('Threshold: {}'.format(threshold), log_file)
    for dtype in dtypes:
        for tool in tools:
            Paths = Path(directory).glob('*' + dtype + '_' + tool + '.tfrecord')
            paths = [str(path) for path in Paths]
            dataset = ds.OrderedDataset(paths, 49, time_window, 100, processed = False)
            result = model.evaluate(dataset, verbose = 0)
            predictions = model.predict(dataset)
            paths = [(path[:-8] + 'csv') for path in paths]
            dataframes = [pd.read_csv(path, header = None, skiprows = 31) for path in paths]
            dataframe = pd.concat(dataframes, ignore_index = True)
            labels = dataframe.to_numpy()
            y = labels[:, 0]
            JTS = labels[:, 2]
            DOB = labels[:, 3]
            print_log('----------------------------------------', log_file)
            print_log((tool + ', ' + dtype), log_file)
            print_log('Accuracy: {:.2}'.format(result[1]), log_file)
            print_log('False positives: {}'.format(int(result[2])), log_file)
            print_log('False negatives: {}'.format(int(result[3])), log_file)
            if dtype == 'collision':
                analyze_collision(threshold, predictions, y, JTS, DOB, log_file)
            else:
                analyze_free(threshold, predictions, y, JTS, DOB, log_file)
            if args.plot != True:
                continue
            length = len(y)
            start = 0
            for i in range(length):
                if y[i] == 1:
                    start = i
                    break
            start = max([(start - 100), 0])
            stop = min([(start + timesteps), length])
            t = range(start, stop)
            plt.figure(figsize = (15, 4.8))
            plt.plot(t, predictions[start:stop, 0], color = 'b', label = 'NN')
            plt.plot(t, y[start:stop], color = 'r', label = 'real')
            plt.plot(t, JTS[start:stop], color = 'k', label = 'JTS')
            plt.plot(t, DOB[start:stop], color = 'y', label = 'DOB')
            plt.xlabel('time')
            plt.ylabel('Collision Probability')
            plt.legend()
            plt.xlim(start, (stop - 1))
            plt.ylim(0, 1.1)
            plt.savefig('/home/dyros/mc_ws/CollisionNet/analysis/' + name1 + '_' + name2 + '_ensemble_' + tool + '_' + dtype + '.png')
            plt.clf()
    log_file.close()