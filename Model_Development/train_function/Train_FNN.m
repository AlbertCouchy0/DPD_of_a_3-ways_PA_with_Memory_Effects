function model=Train_FNN(M,input,output,vec)
%% 预处理
[model_input,model_output] = pre_train(M,input,output,dim);
net = feedforwardnet(vec);
model = train(net, model_input, model_output);
end