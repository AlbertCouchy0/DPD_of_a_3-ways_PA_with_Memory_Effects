function [X,Y]=pre_train(M,x_i,x_q,y_i,y_q,dim)

X = zeros(2*(M + 1), dim-M);
for j = 1:length(y_i) - M
    X(:, j) = [y_i(j : j + M); y_q(j : j + M)];
end
Y = [x_i(M + 1 : end)'; x_q(M + 1 : end)'];

end