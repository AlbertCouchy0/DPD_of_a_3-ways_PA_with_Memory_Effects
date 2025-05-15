function model=Train_FNN_simple(M,input,output,vec)
%% 预处理
% dim = length(input);
dim = 4000;
input=input(20:20+dim,1);
output=output(20:20+dim,1);
[model_input,model_output] = pre_train(M,input,output,dim);

%% 网络设置
net = feedforwardnet(vec);
% 配置训练选项
net.trainFcn = 'trainlm';       % 使用Levenberg-Marquardt算法
net.trainParam.epochs = 4000;   % 最大迭代次数
net.trainParam.max_fail = 6;    % 验证失败次数（类似ValidationPatience）

% 划分训练集和验证集
net.divideFcn = 'divideind'; 
net.divideParam.trainInd = 1:2100;
net.divideParam.valInd = 2101:dim;

model = train(net, model_input, model_output);
end