function model = Transformer_simple(M, dim,input, output)
[model_input, model_output] = pre_train(M, input, output, dim);

%% 构建Transformer网络
net = layerGraph();
tempNet = sequenceInputLayer(2*(M+1),"Name","input");
net = addLayers(net,tempNet);
tempNet = positionEmbeddingLayer(2*(M+1),1024*8,"Name","pos-emb");
net = addLayers(net,tempNet);
tempNet = [
    additionLayer(2,"Name","add")
    selfAttentionLayer(4,128,"Name","selfattention_1")
    functionLayer(@(x)x,"Name",'layer')
    ];
net = addLayers(net,tempNet);
net = connectLayers(net,"input","pos-emb");
net = connectLayers(net,"input","add/in1");
net = connectLayers(net,"pos-emb","add/in2");
% 添加输出层
outputLayers = [
    fullyConnectedLayer(32, "Name", "fc1")
    reluLayer("Name", "fc_relu")
    fullyConnectedLayer(2,"Name","fc2")
    regressionLayer(Name="output")];
net = addLayers(net, outputLayers);
net = connectLayers(net, "layer", "fc1");

%% 显示网络结构
analyzeNetwork(net)

%% 划分训练集与验证集（根据原数据量调整）
numStart = 0;
numTrain = 6000; % 训练样本数
numValid = 7500; % 训练样本数
XTrain = model_input(:, numStart+1:numStart+numTrain);
YTrain = model_output(:, numStart+1:numStart+numTrain);
XVal = model_input(:, numStart+numTrain+1:numStart+numTrain+numValid);
YVal = model_output(:, numStart+numTrain+1:numStart+numTrain+numValid);

%% 调整训练选项（增加梯度裁剪和降低学习率）
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...      % 降低初始学习率
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 300, ...
    'MaxEpochs', 8000, ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 30, ...
    'ValidationPatience', 12, ...         % 增加验证耐心值
    'MiniBatchSize', 128, ...            % 减小批大小
    'GradientThreshold', 1, ...        % 增大梯度阈值
    'Shuffle', 'never', ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'multi-gpu', ...
    'Verbose', 1);

%% 训练模型
model = trainNetwork(XTrain, YTrain, net, options);
end

