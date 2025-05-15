function model=Train_predict(M,dim,input,output)
%% 预训练
[model_input, model_output] = pre_train(M, input, output, dim);

%% 网络构建
net = layerGraph();

% 输入层（维度应为 [特征数×序列长度×样本数]）
inputLayer = sequenceInputLayer(2*(M+1), Name="input", Normalization="rescale-symmetric");
net = addLayers(net, inputLayer);

% 第一卷积块
convBlock1 = [
    convolution1dLayer(5, 64, Name="conv1_1", Padding="causal")
    layerNormalizationLayer(Name="layernorm_1")
    dropoutLayer(0.005,"Name","dropout_1")  % 名称值对格式
    convolution1dLayer(5, 64, Name="conv1_2", Padding="causal")
    layerNormalizationLayer(Name="layernorm_2")
    reluLayer(Name="relu_1")
    dropoutLayer(0.005,"Name","dropout_2")  % 名称值对格式
];
net = addLayers(net, convBlock1);

% 跳跃连接（1x1卷积调整通道数）
skipConv = convolution1dLayer(1, 64, Name="convSkip");
net = addLayers(net, skipConv);

% 残差连接1
net = addLayers(net, additionLayer(2, Name="add_1"));
net = connectLayers(net, "input", "conv1_1");
net = connectLayers(net, "input", "convSkip");
net = connectLayers(net, "dropout_2", "add_1/in1");  % 修正连接点
net = connectLayers(net, "convSkip", "add_1/in2");

% 第二卷积块（膨胀卷积）
convBlock2 = [
    convolution1dLayer(5, 64, Name="conv2_1", DilationFactor=2, Padding="causal")
    layerNormalizationLayer(Name="layernorm_3")
    dropoutLayer(0.005,"Name","dropout_3") % 名称值对格式
    convolution1dLayer(5, 64, Name="conv2_2", DilationFactor=2, Padding="causal")
    layerNormalizationLayer(Name="layernorm_4")
    reluLayer(Name="relu_2")
    dropoutLayer(0.005,"Name","dropout_4")  % 名称值对格式
];
net = addLayers(net, convBlock2);

% 残差连接2
net = addLayers(net, additionLayer(2, Name="add_2"));
net = connectLayers(net, "add_1", "conv2_1");
net = connectLayers(net, "dropout_4", "add_2/in1");
net = connectLayers(net, "add_1", "add_2/in2");

% 第三卷积块（更高膨胀系数）
convBlock3 = [
    convolution1dLayer(5, 64, Name="conv3_1", DilationFactor=4, Padding="causal")
    layerNormalizationLayer(Name="layernorm_5")
    dropoutLayer(0.005,"Name","dropout_5")  % 名称值对格式
    convolution1dLayer(5, 64, Name="conv3_2", DilationFactor=4, Padding="causal")
    layerNormalizationLayer(Name="layernorm_6")
    reluLayer(Name="relu_3")
    dropoutLayer(0.005,"Name","dropout_6")  % 名称值对格式
];
net = addLayers(net, convBlock3);

% 残差连接3
net = addLayers(net, additionLayer(2, Name="add_3"));
net = connectLayers(net, "add_2", "conv3_1");
net = connectLayers(net, "dropout_6", "add_3/in1");
net = connectLayers(net, "add_2", "add_3/in2");

% 第四卷积块（最高膨胀系数）
convBlock4 = [
    convolution1dLayer(5, 64, Name="conv4_1", DilationFactor=8, Padding="causal")
    layerNormalizationLayer(Name="layernorm_7")
    dropoutLayer(0.005,"Name","dropout_7")  % 名称值对格式
    convolution1dLayer(5, 64, Name="conv4_2", DilationFactor=8, Padding="causal")
    layerNormalizationLayer(Name="layernorm_8")
    reluLayer(Name="relu_4")
    dropoutLayer(0.005,"Name","dropout_8")  % 名称值对格式
];
net = addLayers(net, convBlock4);

% 最终残差连接和输出层
net = addLayers(net, additionLayer(2, Name="add_4"));
net = connectLayers(net, "add_3", "conv4_1");
net = connectLayers(net, "dropout_8", "add_4/in1");
net = connectLayers(net, "add_3", "add_4/in2");

% 输出层（回归任务）
outputLayers = [
    fullyConnectedLayer(2, Name="fc")  % 输出维度为1
    regressionLayer(Name="output")     % 回归层
];
net = addLayers(net, outputLayers);
net = connectLayers(net, "add_4", "fc");

% 分析网络结构
analyzeNetwork(net);

%% 划分训练集与验证集（根据原数据量调整）
numTrain = 6000; % 训练样本数
XTrain = model_input(:, 1:numTrain);
YTrain = model_output(:, 1:numTrain);
XVal = model_input(:, numTrain+1:end);
YVal = model_output(:, numTrain+1:end);

%% 配置训练选项
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.003, ...  % 通常比Adam稍小
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 400, ...
    'MaxEpochs', 16000, ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 6, ...
    'MiniBatchSize', 256, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'never', ... % 若数据为时序需关闭Shuffle
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'multi-gpu', ... % 使用GPU加速
    'Verbose', 1);

%% 训练网络
model = trainNetwork(XTrain, YTrain, net, options);

end



% function model=Train_predict(M,dim,input,output)
% %% 预训练（无需修改）
% [model_input,model_output] = pre_train(M,input,output,dim);
% 
% % 网络设置
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
% 
% % layers = [
% %     sequenceInputLayer(2*(M+1), 'Name', 'input')  % 输入形状 [8, N]
% %     convolution1dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1')  % 核大小3，64个滤波器
% %     batchNormalizationLayer('Name', 'bn1')
% %     reluLayer('Name', 'relu1')
% %     lstmLayer(64, 'OutputMode', 'sequence', 'Name', 'lstm1')
% %     tanhLayer("Name", "tanh1")
% %     fullyConnectedLayer(2, 'Name', 'fc')
% %     regressionLayer('Name', 'output')
% % ];
% % 
% % layers = [
% %     sequenceInputLayer(2*(M+1), 'Name', 'input')  % 输入形状 [8, N]
% %     gruLayer(128, 'OutputMode', 'sequence', 'Name', 'gru1')
% %     tanhLayer("Name", "tanh1")
% %     fullyConnectedLayer(2, 'Name', 'fc')
% %     regressionLayer('Name', 'output')
% % ];
% 
% %% 划分训练集与验证集（根据原数据量调整）
% numTrain = 6000; % 训练样本数
% XTrain = model_input(:, 1:numTrain);
% YTrain = model_output(:, 1:numTrain);
% XVal = model_input(:, numTrain+1:end);
% YVal = model_output(:, numTrain+1:end);
% 
% %% 配置训练选项
% options = trainingOptions('adam', ...% 'SquaredGradientDecayFactor', 0.9, ...
%     'InitialLearnRate', 0.001, ...  % 通常比Adam稍小
%     'LearnRateSchedule', 'piecewise', ...
%     'LearnRateDropFactor', 0.5, ...
%     'LearnRateDropPeriod', 400, ...
%     'MaxEpochs', 4000, ...
%     'ValidationData', {XVal, YVal}, ...
%     'ValidationFrequency', 50, ...
%     'ValidationPatience', 6, ...
%     'MiniBatchSize', 512, ...
%     'GradientThreshold', 1, ...
%     'Shuffle', 'never', ... % 若数据为时序需关闭Shuffle
%     'Plots', 'training-progress', ...
%     'ExecutionEnvironment', 'multi-gpu', ... % 使用GPU加速
%     'Verbose', 1);
% analyzeNetwork(layers);
% 
% %% 训练网络
% model = trainNetwork(XTrain, YTrain, layers, options);
% 
% end