import tensorflow as tf
import pandas as pd
import numpy as np
import dataset_experimental as ds
import argparse
import os
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--model1', type = str, default = 'model1')
parser.add_argument('--model2', type = str, default = 'model2')
args = parser.parse_args()
model1 = args.model1
model2 = args.model2

# Parameter
time_window = 32
num_data = 49
threshold = 0.99

if tf.__version__.startswith('2'):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
else:
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config = tf_config)
    tf.compat.v1.keras.backend.set_session(sess)

# Model 1
model1 = tf.keras.models.load_model(('/home/dyros/mc_ws/CollisionNet/model/' + model1 + '.h5'), compile = True)

# Model 2
model2 = tf.keras.models.load_model(('/home/dyros/mc_ws/CollisionNet/model/' + model2 + '.h5'), compile = True)

# Datasets
p = Path('/home/dyros/mc_ws/CollisionNet/data/test')
datafiles = [x for x in p.iterdir() if x.suffix == '.tfrecord']

for datafile in datafiles:
    # Collision Data
    if 'Collision' in str(datafile):
        dataset = ds.OneDataset(str(datafile), num_data, time_window, 50)
        TestData = pd.read_csv(datafile.with_suffix('.csv'), header = None, skiprows = (lambda x: x < (time_window - 1))).to_numpy()
        Y_Test = TestData[:, :2]
        JTS = TestData[:, 2]
        DOB = TestData[:, 3]
        hypo1  =  model1.predict(dataset)
        hypo2  =  model2.predict(dataset)

        t = np.arange(0, (0.001 * len(JTS)), 0.001)
        collision_pre = 0
        collision_cnt = 0
        collision_time = 0
        detection_time_NN = []
        detection_time_JTS = []
        detection_time_DoB = []
        collision_status = False
        NN_detection = False
        JTS_detection = False
        DoB_detection = False
        collision_fail_cnt_NN = 0
        collision_fail_cnt_JTS = 0
        collision_fail_cnt_DoB = 0

        for i in range(len(JTS)):
            if (Y_Test[i, 0] == 1 and collision_pre == 0):
                collision_cnt = collision_cnt + 1
                collision_time = t[i]
                collision_status = True
                NN_detection = False
                JTS_detection = False
                DoB_detection = False
            
            if (collision_status == True and NN_detection == False):
                if(hypo1[i, 0] > threshold and hypo2[i, 0] > threshold):
                    NN_detection = True
                    detection_time_NN.append(t[i] - collision_time)

            if (collision_status == True and JTS_detection == False):
                if(JTS[i] == 1):
                    JTS_detection = True
                    detection_time_JTS.append(t[i] - collision_time)
            
            if (collision_status == True and DoB_detection == False):
                if(DOB[i] == 1):
                    DoB_detection = True
                    detection_time_DoB.append(t[i] - collision_time)

            if (Y_Test[i, 0] == 0 and collision_pre == 1):
                collision_status = False
                if(NN_detection == False):
                    detection_time_NN.append(0.0)
                    collision_fail_cnt_NN = collision_fail_cnt_NN + 1
                if(JTS_detection == False):
                    detection_time_JTS.append(0.0)
                    collision_fail_cnt_JTS = collision_fail_cnt_JTS + 1
                if(DoB_detection == False):
                    detection_time_DoB.append(0.0)
                    collision_fail_cnt_DoB = collision_fail_cnt_DoB + 1
            collision_pre = Y_Test[i, 0]
        print('----------------------------------------')
        #print('Tool ', Tool_list[tool_idx],' ', DataType_list[data_idx])
        print('Total collision: ', collision_cnt)
        print('NN Failure: ', collision_fail_cnt_NN)
        print('JTS Failure: ', collision_fail_cnt_JTS)
        print('DOB Failure: ', collision_fail_cnt_DoB)
        print('NN Detection Time: ', sum(detection_time_NN) / (collision_cnt - collision_fail_cnt_NN))
        print('JTS Detection Time: ', sum(detection_time_JTS) / (collision_cnt - collision_fail_cnt_JTS))
        print('DOB Detection Time: ', sum(detection_time_DoB) / (collision_cnt - collision_fail_cnt_DoB))

    # Free Data
    else :
        dataset = ds.OneDataset(str(datafile), num_data, time_window, 50)
        TestDataFree = pd.read_csv(datafile.with_suffix('.csv'), header = None, skiprows = (lambda x: x < (time_window - 1))).to_numpy()
        Y_TestFree = TestData[:, :2]
        JTSFree = TestDataFree[:, 2]
        DOBFree = TestDataFree[:, 3]
        hypofree1  =  model1.predict(dataset)
        hypofree2  =  model2.predict(dataset)

        t_free = np.arange(0, (0.001 * len(JTSFree)), 0.001)
        NN_FP_time = []
        NN_FP = 0
        JTS_FP_time = []
        JTS_FP = 0
        DOB_FP_time = []
        DOB_FP = 0
        for j in range(len(Y_TestFree)):
            if (hypofree1[j, 0] > threshold and hypofree2[j, 0] > threshold  and np.equal(np.argmax(Y_TestFree[j, :]), 1)):
                NN_FP_time.append(t_free[j])
                NN_FP = NN_FP + 1
            if (JTSFree[j] == 1 and np.equal(np.argmax(Y_TestFree[j, :]), 1)):
                JTS_FP_time.append(t_free[j])
                JTS_FP = JTS_FP + 1
            if (DOBFree[j] == 1 and np.equal(np.argmax(Y_TestFree[j, :]), 1)):
                DOB_FP_time.append(t_free[j])
                DOB_FP = DOB_FP + 1
        print('----------------------------------------')
        #print('Tool ', Tool_list[tool_idx],' ', DataType_list[data_idx])
        print("NN FP Time: ")
        for k in range(NN_FP - 1):
            del_time = abs(NN_FP_time[k + 1]- NN_FP_time[k])
            if(del_time > 0.5):
                print(del_time)
        print("JTS FP Time: ")
        for k in range(JTS_FP - 1):
            del_time = abs(JTS_FP_time[k + 1]- JTS_FP_time[k])
            if(del_time > 0.5):
                print(del_time)
        print("DOB FP Time: ")
        for k in range(DOB_FP - 1):
            del_time = abs(DOB_FP_time[k + 1]- DOB_FP_time[k])
            if(del_time > 0.5):
                print(del_time)