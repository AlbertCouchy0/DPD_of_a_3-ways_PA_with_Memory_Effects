function output_sim=DPD_predict(input,M,path)
dim=length(input);
input_vector1=zeros(2*(M+1),dim);

for i=1:M+1
    input_vector1(i,:) = real(Pack( Delay(input,M+1-i) ));
    input_vector1(i+M+1,:) = imag(Pack( Delay(input,M+1-i) ));
end

data = load(path);
DPD_output_m = predict(data.model,input_vector1);
DPD_output_m = double(DPD_output_m);
DPD_output = DPD_output_m(1,:) + 1j * DPD_output_m(2,:);

% 设置输出
output_sim = UnPack(DPD_output);
end