function model=Train_FNN(M,dim,input,output,vec)
net = feedforwardnet(vec);
[model_input,model_output] = pre_train(M,real(input),imag(input),real(output),imag(output),dim);
model = train(net, model_input, model_output);
end