function model=Train_Whole(M,dim,input,output)
%% 预训练（无需修改）
[model_input,model_output] = pre_train(M,input,output,dim);

%% 网络设置
% layers = [
%     sequenceInputLayer(2*(M+1), "Name", "input")
%     fullyConnectedLayer(128, "Name", "fc1")
%     tanhLayer("Name", "tanh1")
%     fullyConnectedLayer(64, "Name", "fc2")
%     tanhLayer("Name", "tanh2")
%     fullyConnectedLayer(32, "Name", "fc3")
%     tanhLayer("Name", "tanh3")
%     fullyConnectedLayer(16, "Name", "fc4")
%     tanhLayer("Name", "tanh4")
%     fullyConnectedLayer(2, "Name", "output")
%     regressionLayer("Name", "regressionOutput")
% ];

% layers = [
%     sequenceInputLayer(2*(M+1), 'Name', 'input')  % 输入形状 [2*(M+1), N]
%     convolution1dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1')  % 核大小3，64个滤波器
%     batchNormalizationLayer('Name', 'bn1')
%     reluLayer('Name', 'relu1')
%     lstmLayer(64, 'OutputMode', 'sequence', 'Name', 'lstm1')
%     tanhLayer("Name", "tanh1")
%     fullyConnectedLayer(2, 'Name', 'fc')
%     regressionLayer('Name', 'output')
% ];

layers = [
    sequenceInputLayer(2*(M+1), 'Name', 'input')  % 输入形状 [2*(M+1), N]
    gruLayer(64, 'OutputMode', 'sequence', 'Name', 'gru1')
    gruLayer(64, 'OutputMode', 'sequence', 'Name', 'gru2')
    tanhLayer("Name", "tanh1")
    fullyConnectedLayer(16, 'Name', 'fc1')
    tanhLayer("Name", "tanh2")
    fullyConnectedLayer(2, 'Name', 'fc2')
    regressionLayer('Name', 'output')
];

%% 划分训练集与验证集（根据原数据量调整）
numTrain = 5500; % 训练样本数
XTrain = model_input(:, 1:numTrain);
YTrain = model_output(:, 1:numTrain);
XVal = model_input(:, numTrain+1:end);
YVal = model_output(:, numTrain+1:end);

%% 配置训练选项
options = trainingOptions('adam', ...% 'SquaredGradientDecayFactor', 0.9, ...
    'InitialLearnRate', 0.003, ...  % 通常比Adam稍小
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 500, ...
    'MaxEpochs', 4000, ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 8, ...
    'MiniBatchSize', 512, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'never', ... % 若数据为时序需关闭Shuffle
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'multi-gpu', ... % 使用GPU加速
    'Verbose', 1);
analyzeNetwork(layers);

%% 训练网络
model = trainNetwork(XTrain, YTrain, layers, options);

end