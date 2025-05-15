function model=Train_net(M,input,output,vec,regularizationCoeff)
%% 预处理
dim = length(input);
% dim = 4000;
% input=input(20:20+dim,1);
% output=output(20:20+dim,1);
div = 2100;
[model_input,model_output] = pre_train(0,input,output,dim);
% 转换为细胞数组（每个时间步为2×1列向量）
model_input = num2cell(model_input, 1);  % 转为1×N细胞数组
model_output = num2cell(model_output, 1);

% 创建时间延迟网络
inputDelays = 1:M;
net = timedelaynet(inputDelays, vec);

% 准备训练数据
[Xs, Xi, Ai, Ts] = preparets(net, model_input, model_output);

% 配置训练选项
net.trainFcn = 'trainlm';       % 使用Levenberg-Marquardt算法
net.trainParam.epochs = 4000;   % 最大迭代次数
net.trainParam.max_fail = 10;    % 验证失败次数（类似ValidationPatience）

% 训练网络
model = train(net, Xs, Ts, Xi, Ai);
% model = train(net, Xs, Ts, Xi, Ai,'UseGPU', 'yes');
end




