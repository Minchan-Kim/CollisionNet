import pandas as pd
import numpy as np


MaxTrainingData = np.array([
    25,  # 시간 
    300,300,200,100,100,50,  # 전류기반 토크 
    3.14,3.14,1.57,3.14,3.14,3.14,  # 엔코더 각도 
    1.57,1.57,1.57,1.57,1.57,1.57,  # 엔코더 각속도
    3.14,3.14,1.57,3.14,3.14,3.14,   # 목표 각도
    1.57,1.57,1.57,1.57,1.57,1.57,  # 목표 각속도
    300,300,200,100,100,50,  # 동적 토크
    3.14,3.14,1.57,3.14,3.14,3.14,  # 절대엔코더 각도
    1.57,1.57,1.57,1.57,1.57,1.57,  # 절대엔코더 각속도 
    -35.8125978300000,-61.6178708300000,-35.7297740300000,-15.0661117900000,-13.1961032600000,-12.8466518500000,  # 추정 마찰력
    60,60,60,60,60,60,  # 온도
    30,30,30,  # 말단 가속도 
    0,0,0,  # 스위치, JTS충돌, 전류충돌 
    300,300,200,100,100,50]) # JTS기반 토크 

MinTrainingData = np.array([
    25,  # 시간 
    -300,-300,-200,-100,-100,-50,  # 전류기반 토크 
    -3.14,-3.14,-1.57,-3.14,-3.14,-3.14,  # 엔코더 각도 
    -1.57,-1.57,-1.57,-1.57,-1.57,-1.57,  # 엔코더 각속도
    -3.14,-3.14,-1.57,-3.14,-3.14,-3.14,  # 목표 각도
    -1.57,-1.57,-1.57,-1.57,-1.57,-1.57,  # 목표 각속도
    -300,-300,-200,-100,-100,-50,  # 동적 토크
    -3.14,-3.14,-1.57,-3.14,-3.14,-3.14,  # 절대엔코더 각도
    -1.57,-1.57,-1.57,-1.57,-1.57,-1.57,  # 절대엔코더 각속도 
    -35.8125978300000,-61.6178708300000,-35.7297740300000,-15.0661117900000,-13.1961032600000,-12.8466518500000,  # 추정 마찰력
    5.0,5.0,5.0,5.0,5.0,5.0,  # 온도
    -30,-30,-30,  # 말단 가속도 
    0,0,0,  # 스위치, JTS충돌, 전류충돌 
    -300,-300,-200,-100,-100,-50]) # JTS기반 토크 

# Max and Min of Explicit Input
MaxDeltaCurrentDyna = 1.0e+02 * np.array([1.446886991650000, 1.668623672100000, 0.975043730100000, 0.366971076220000, 0.388747072440000, 0.155852918317100])
MinDeltaCurrentDyna = 1.0e+02 * np.array([-1.707753304150000, -1.653056572400000, -0.922283913690000, -0.429723359370000, -0.222735385400000, -0.129340748762640])
MaxDeltaQdQ = np.array([0.012106301100000, 0.009996968500000, 0.008732073000000, 0.010951596900000, 0.009698796749000, 0.007999126000000])
MinDeltaQdQ = np.array([-0.011341421000000, -0.007788007100000, -0.011375057900000, -0.011238997400000, -0.010040846200000, -0.009056478100000])
MaxDeltaQdotdQdot = np.array([0.030471368000000, 0.032628351300000, 0.032703119000000, 0.055313989900000, 0.041539804000000, 0.023768547000000])
MinDeltaQdotdQdot = np.array([-0.037924034800000, -0.033170233000000, -0.037959729000000, -0.044833251500000, -0.044674244900000, -0.023149635200000])
MaxDeltaThetaQ = np.array([0.001582920000000, 0.002739420600000, 0.002450675600000, 0.003926920000000, 0.000942399100000, 0.001213046000000])
MinDeltaThetaQ = np.array([-0.002181576000000, -0.003093831000000, -0.002274677500000, -0.001609872000000, -0.001274405000000, -0.001873778000000])
MaxAcc = 10
MinAcc = 0

def normalize_data(data, i, j):
    return (2 * (data[i:j] - MinTrainingData[i:j]) / (MaxTrainingData[i:j] - MinTrainingData[i:j]) - 1)

def Normalize(src, dest, write, time_window, frequency, tools, dtype):
    Data = pd.read_csv(src, header = None, delim_whitespace = True).to_numpy()
    size = (Data.shape)[0]
    ProcessedData = np.zeros((size, 53), dtype = Data.dtype)
    start = 0

    for i in range(size):
        if (i > 0) and (round((Data[i, 0] - Data[(i - 1), 0]), 3) != (1 / float(frequency))):
            if (i - start) >= time_window:
                write(ProcessedData[start:i, :], src, dest, time_window, tools, dtype)
            start = i
        data = Data[i, :]
        q = normalize_data(data, 7, 13)
        q_dot = normalize_data(data, 13, 19)
        q_r = normalize_data(data, 19, 25)
        q_r_dot = normalize_data(data, 25, 31)
        e = (2 * (data[19:25] - data[7:13] - MinDeltaQdQ) / (MaxDeltaQdQ - MinDeltaQdQ) - 1)
        e_dot = (2 * (data[25:31] - data[13:19] - MinDeltaQdotdQdot) / (MaxDeltaQdotdQdot - MinDeltaQdotdQdot) - 1)
        tau_m = normalize_data(data, 1, 7)
        tau_dyn = normalize_data(data, 31, 37)
        a = ((np.linalg.norm(data[61:64]) - MinAcc) / (MaxAcc - MinAcc))
        x = np.concatenate((q, q_dot, q_r, q_r_dot, e, e_dot, tau_m, tau_dyn, a), axis = None)
        y = np.array([data[64], (1 - data[64])])
        z = np.array([data[65], data[66]])
        ProcessedData[i, :-4] = x
        ProcessedData[i, -4:-2] = y
        ProcessedData[i, -2:] = z
    if (size - start) >= time_window:
        write(ProcessedData[start:size, :], src, dest, time_window, tools, dtype)