function DPD_coefficient=Train_Voltrra(M,N,dim,input,output)
output=output/max(abs(output));
input=input/max(abs(input));
DPD_input_matrix = zeros(dim, (M+1)*N);
for i=1:M+1
    for j= 1:N
        DPD_input_matrix(:,N*(i-1)+j) = Delay(output,i-1).*((Delay(output,i-1)).^(2*j-2));
    end
end
DPD_coefficient=DPD_input_matrix\input(:);
end