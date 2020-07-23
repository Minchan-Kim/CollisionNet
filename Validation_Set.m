clear all
clc
format long

Original_hz = 1000;
Reduced_hz = 100;
Reduced_interval = Original_hz / Reduced_hz;

DoosanMSeries_RoboticsToolBox_Simulator;
cd ValidationSet

%% Collision
ValidationData = load('robot1/5_01kg/1/DRCL_Data_9.txt');
ReducedDataValidation = zeros(fix(size(ValidationData, 1) / Reduced_interval), size(ValidationData, 2));
for row = 1:size(ReducedDataValidation, 1)
    ReducedDataValidation(row, :) = ValidationData(Reduced_interval * (row - 1) + 1, :);
end

% 가속도 보상 & 동적토크 모터 관성 추가
AngleM = ReducedDataValidation(:, 8:13);
VelD = ReducedDataValidation(:, 26:31);
ToolAccM = ReducedDataValidation(:, 62:64);
TorqueDyn = ReducedDataValidation(:, 32:37);

ToolGrav = zeros(size(ReducedDataValidation, 1), 3);
ToolAccD = zeros(size(ReducedDataValidation, 1), 3);
BaseAccD = zeros(size(ReducedDataValidation, 1), 6);
AccD = zeros(size(ReducedDataValidation, 1), 6);
for m = 1:size(ReducedDataValidation, 1)
    R_T = bot.fkine(AngleM(m, :)).R()';
    if m == 1
        AccD(m, :) = (VelD((m + 1), :) - VelD(m, :)) * Reduced_hz;
    else
        AccD(m, :) = (VelD(m, :) - VelD((m - 1), :)) * Reduced_hz;
    end
    ToolGrav(m, :) = (R_T * [0.0; 0.0; -9.80])';
    BaseAccD(m, :) = bot.jacob0(AngleM(m, :)) * AccD(m, :)' + bot.jacob_dot(AngleM(m, :), VelD(m, :));
    ToolAccD(m, :) = (R_T * BaseAccD(m, 1:3)')';
    if mod(m, 10000) == 0
        m
    end
end
for j = 1:6
    TorqueDyn(:, j) = TorqueDyn(:, j) + bot.links(j).Jm * bot.links(j).G * bot.links(j).G * AccD(:, j);
end
ReducedDataValidation(:, 62:64) = ToolAccM + ToolGrav - ToolAccD;
ReducedDataValidation(:, 32:37) = TorqueDyn;
save('robot1/5_01kg/1/Reduced_DRCL_ValidationData.txt', 'ReducedDataValidation', '-ascii', '-double', '-tabs')
cd ..
return
%% Data Process For normalization

process_hz = 100;
num_data_type = 9;
num_input = 6 * (num_data_type - 1) + 1;
time_window = 16;

MaxTrainingData = ...
[25, ... % 시간 
300,300,200,100,100,50, ... % 전류기반 토크 
3.14,3.14,1.57,3.14,3.14,3.14, ... % 엔코더 각도 
1.57,1.57,1.57,1.57,1.57,1.57, ... % 엔코더 각속도
3.14,3.14,1.57,3.14,3.14,3.14, ...  % 목표 각도
1.57,1.57,1.57,1.57,1.57,1.57, ... % 목표 각속도
300,300,200,100,100,50, ... % 동적 토크
3.14,3.14,1.57,3.14,3.14,3.14, ... % 절대엔코더 각도
1.57,1.57,1.57,1.57,1.57,1.57, ... % 절대엔코더 각속도 
-35.8125978300000,-61.6178708300000,-35.7297740300000,-15.0661117900000,-13.1961032600000,-12.8466518500000, ... % 추정 마찰력
60,60,60,60,60,60, ... % 온도
30,30,30, ... % 말단 가속도 
0,0,0, ... % 스위치, JTS충돌, 전류충돌 
300,300,200,100,100,50]; % JTS기반 토크 

MinTrainingData = ...
[25, ... % 시간 
-300,-300,-200,-100,-100,-50, ... % 전류기반 토크 
-3.14,-3.14,-1.57,-3.14,-3.14,-3.14, ... % 엔코더 각도 
-1.57,-1.57,-1.57,-1.57,-1.57,-1.57, ... % 엔코더 각속도
-3.14,-3.14,-1.57,-3.14,-3.14,-3.14, ... % 목표 각도
-1.57,-1.57,-1.57,-1.57,-1.57,-1.57, ... % 목표 각속도
-300,-300,-200,-100,-100,-50, ... % 동적 토크
-3.14,-3.14,-1.57,-3.14,-3.14,-3.14, ... % 절대엔코더 각도
-1.57,-1.57,-1.57,-1.57,-1.57,-1.57, ... % 절대엔코더 각속도 
-35.8125978300000,-61.6178708300000,-35.7297740300000,-15.0661117900000,-13.1961032600000,-12.8466518500000, ... % 추정 마찰력
5.0,5.0,5.0,5.0,5.0,5.0, ... % 온도
-30,-30,-30, ... % 말단 가속도 
0,0,0, ... % 스위치, JTS충돌, 전류충돌 
-300,-300,-200,-100,-100,-50]; % JTS기반 토크 

MaxCurrentDyna = 1.0e+02 * [1.446886991650000   1.668623672100000   0.975043730100000   0.366971076220000   0.388747072440000   0.155852918317100];
MinCurrentDyna = 1.0e+02 * [-1.707753304150000  -1.653056572400000  -0.922283913690000  -0.429723359370000  -0.222735385400000  -0.129340748762640];
MaxDeltaQdQ = [0.012106301100000   0.009996968500000   0.008732073000000   0.010951596900000   0.009698796749000   0.007999126000000];
MinDeltaQdQ = [-0.011341421000000  -0.007788007100000  -0.011375057900000  -0.011238997400000  -0.010040846200000  -0.009056478100000];
MaxDeltaQdotdQdot = [0.030471368000000   0.032628351300000   0.032703119000000   0.055313989900000   0.041539804000000   0.023768547000000];
MinDeltaQdotdQdot = [-0.037924034800000  -0.033170233000000  -0.037959729000000  -0.044833251500000  -0.044674244900000  -0.023149635200000];
MaxDeltaThetaQ = [0.001582920000000   0.002739420600000   0.002450675600000   0.003926920000000   0.000942399100000   0.001213046000000];
MinDeltaThetaQ = [-0.002181576000000  -0.003093831000000  -0.002274677500000  -0.001609872000000  -0.001274405000000  -0.001873778000000];
MaxAcc = 10;
MinAcc = 0;

ValidationProcessData = zeros(size(ReducedDataValidation, 1), (num_input + 2));
Labels = zeros(size(ReducedDataValidation, 1), 4);

NormalizeData = @(rawdata, i, j) (2 .* (rawdata(i:j) - MinTrainingData(i:j)) ./ (MaxTrainingData(i:j) - MinTrainingData(i:j)) - 1);
start = 1;
file_idx = 1;
path = '/home/dyros/mc_ws/CollisionNet/data/16/validation';
for k = 1:size(ReducedDataValidation, 1)
    % Check time stamp
    if k > 1
        dt_data = round(ReducedDataValidation(k, 1) - ReducedDataValidation((k - Reduced_hz / process_hz), 1), 3);
        if dt_data ~= (1 / process_hz)
            if (k - start) >= time_window
                dlmwrite(strcat(path, '/temp', int2str(file_idx), '.csv'), ValidationProcessData(start:(k - 1), :), 'precision', '%.8e');
                csvwrite(strcat(path, '/validation', int2str(file_idx), '.csv'), Labels(start:(k - 1), :));
                file_idx = file_idx + 1;
            end
            start = k;
        end
    end
    
    ValidationProcessData(k, 1:6) = NormalizeData(ReducedDataValidation(k, :), 8, 13); % q
    ValidationProcessData(k, 7:12) = NormalizeData(ReducedDataValidation(k, :), 14, 19); % q_dot
    ValidationProcessData(k, 13:18) = NormalizeData(ReducedDataValidation(k, :), 20, 25); % q_r
    ValidationProcessData(k, 19:24) = NormalizeData(ReducedDataValidation(k, :), 26, 31); % q_r_dot
    ValidationProcessData(k, 25:30) = (2 * (ReducedDataValidation(k, 20:25) - ReducedDataValidation(k, 8:13) - MinDeltaQdQ) / (MaxDeltaQdQ - MinDeltaQdQ) - 1); % e
    ValidationProcessData(k, 31:36) = (2 * (ReducedDataValidation(k, 26:31) - ReducedDataValidation(k, 14:19) - MinDeltaQdotdQdot) / (MaxDeltaQdotdQdot - MinDeltaQdotdQdot) - 1); % e_dot
    ValidationProcessData(k, 37:42) = NormalizeData(ReducedDataValidation(k, :), 2, 7); % tau_m
    ValidationProcessData(k, 43:48) = NormalizeData(ReducedDataValidation(k, :), 32, 37); % tau_dyn
    ValidationProcessData(k, 49) = (norm(ReducedDataValidation(k, 62:64)) - MinAcc) / (MaxAcc - MinAcc); % end effector
    ValidationProcessData(k, 50) = ReducedDataValidation(k, 65); % label
    ValidationProcessData(k, 51) = (1 - ReducedDataValidation(k, 65)); % ~label
    Labels(k, 1) = ReducedDataValidation(k, 65); % label
    Labels(k, 2) = (1 - ReducedDataValidation(k, 65)); % ~label
    Labels(k, 3) = ReducedDataValidation(k, 66); % JTS
    Labels(k, 4) = ReducedDataValidation(k, 67); % DOB
end
last = size(ReducedDataValidation, 1);
if (last - start + 1) >= time_window
    dlmwrite(strcat(path, '/temp', int2str(file_idx), '.csv'), ValidationProcessData(start:last, :), 'precision', '%.8e');
    csvwrite(strcat(path, '/validation', int2str(file_idx), '.csv'), Labels(start:last, :));
end
disp('============================')
disp(size(ReducedDataValidation, 1))
clear ReducedDataValidation;
cd ..
