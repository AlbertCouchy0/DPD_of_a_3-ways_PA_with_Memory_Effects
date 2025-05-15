function output_sim=DPD_net(input,M,path)
%% 预处理
input_vec(:,1)=real(input);
input_vec(:,2)=imag(input);
input_cell = num2cell(input_vec.', 1);  % 转为1×N细胞数组


data = load(path);
DPD_output_m = cell2mat(data.model(input_cell(M+1:end),input_cell(1:M),{}));
DPD_output_m = double(DPD_output_m);
DPD_output = DPD_output_m(1,:)+1j*DPD_output_m(2,:);

% 设置输出
output_sim=UnPack(DPD_output);
end