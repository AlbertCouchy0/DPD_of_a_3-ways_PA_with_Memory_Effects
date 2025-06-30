function model = Transformer_complex(M, dim,input, output)
[model_input, model_output] = pre_train(M, input, output, dim);

%% 修正后的网络结构
%% 构建Transformer网络
net = layerGraph();
tempNet = sequenceInputLayer(2*(M+1),"Name","input");
net = addLayers(net,tempNet);
tempNet = positionEmbeddingLayer(2*(M+1),4096*2,"Name","pos-emb");
net = addLayers(net,tempNet);
tempNet = [
    additionLayer(2,"Name","add")
    selfAttentionLayer(8,256,"Name","selfattention_1") % 第1层
    layerNormalizationLayer("Name", "ln_attn_1")
    fullyConnectedLayer(1024, "Name", "ffn_fc1_1")
    reluLayer("Name", "ffn_relu_1")
    dropoutLayer(0.1, "Name", "dropout_1")
    fullyConnectedLayer(256, "Name", "ffn_fc2_1") 
    layerNormalizationLayer("Name", "ln_ffn_1")
    selfAttentionLayer(8,256,"Name","selfattention_2") % 第2层
    layerNormalizationLayer("Name", "ln_attn_2")
    fullyConnectedLayer(1024, "Name", "ffn_fc1_2")
    reluLayer("Name", "ffn_relu_2")
    dropoutLayer(0.1, "Name", "dropout_2")
    fullyConnectedLayer(256, "Name", "ffn_fc2_2") 
    layerNormalizationLayer("Name", "ln_ffn_2")
    selfAttentionLayer(8,256,"Name","selfattention_3") % 第3层
    layerNormalizationLayer("Name", "ln_attn_3")
    fullyConnectedLayer(1024, "Name", "ffn_fc1_3")
    reluLayer("Name", "ffn_relu_3")
    dropoutLayer(0.1, "Name", "dropout_3")
    fullyConnectedLayer(256, "Name", "ffn_fc2_3") 
    layerNormalizationLayer("Name", "ln_ffn_3")
    selfAttentionLayer(8,256,"Name","selfattention_4") % 第4层
    layerNormalizationLayer("Name", "ln_attn_4")
    fullyConnectedLayer(1024, "Name", "ffn_fc1_4")
    reluLayer("Name", "ffn_relu_4")
    dropoutLayer(0.1, "Name", "dropout_4")
    fullyConnectedLayer(256, "Name", "ffn_fc2_4") 
    layerNormalizationLayer("Name", "ln_ffn_4")
    functionLayer(@(x)x,"Name",'layer')
    ];
net = addLayers(net,tempNet);
net = connectLayers(net,"input","pos-emb");
net = connectLayers(net,"input","add/in1");
net = connectLayers(net,"pos-emb","add/in2");
% 添加输出层
outputLayers = [
    fullyConnectedLayer(128, "Name", "fc1")
    reluLayer("Name", "fc_relu")
    fullyConnectedLayer(2,"Name","fc2")
    regressionLayer(Name="output")];
net = addLayers(net, outputLayers);
net = connectLayers(net, "layer", "fc1");


%% 显示网络结构
analyzeNetwork(net)

%% 划分训练集与验证集（根据原数据量调整）
numTrain = 6000; % 训练样本数
XTrain = model_input(:, 1:numTrain);
YTrain = model_output(:, 1:numTrain);
XVal = model_input(:, numTrain+1:end);
YVal = model_output(:, numTrain+1:end);

%% 调整训练选项（增加梯度裁剪和降低学习率）
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...      % 降低初始学习率
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 200, ...
    'MaxEpochs', 8000, ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 20, ...
    'ValidationPatience', 9, ...         % 增加验证耐心值
    'MiniBatchSize', 128, ...            % 减小批大小
    'GradientThreshold', 1, ...        % 增大梯度阈值
    'Shuffle', 'never', ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'multi-gpu', ...
    'Verbose', 1);

%% 训练模型
model = trainNetwork(XTrain, YTrain, net, options);
end

