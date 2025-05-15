function output=DPD_Voltrra(input,M,N,path)
dim = length(input);
C0=2.001251897764724;
C = load(path);
% 数据处理
input_nor = input/C0;
output_nor = zeros(dim,1);
for i=1:M+1
    for j= 1:N
        output_nor = output_nor + C.DPD_coefficient(N*(i-1)+j)*Delay(input_nor,i-1).*((Delay(input_nor,i-1)).^(2*j-2));
    end
end
output=output_nor*C0;

end