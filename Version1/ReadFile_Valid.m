function [Data_matrix,dim]=ReadFile_Valid()
% Data_matrix每一列对应的是
% Data_matrix=[Sys_x_i, Sys_x_q,...
%              PA1_y_i, PA1_y_q,...
%              PA2_y_i, PA2_y_q,...
%              PA3_y_i, PA3_y_q,...
%              Sys_y_i, Sys_y_q];
%% 读文件
fileID = fopen('Data_valid\Sys_input_i.txt'); C = textscan(fileID, '%f'); Sys_input_i = C{1, 1}; fclose(fileID);
fileID = fopen('Data_valid\Sys_input_q.txt'); C = textscan(fileID, '%f'); Sys_input_q = C{1, 1}; fclose(fileID);

fileID = fopen('Data_valid\PA1_output_i.txt'); C = textscan(fileID, '%f'); PA1_output_i = C{1, 1}; fclose(fileID);
fileID = fopen('Data_valid\PA1_output_q.txt'); C = textscan(fileID, '%f'); PA1_output_q = C{1, 1}; fclose(fileID);
fileID = fopen('Data_valid\PA2_output_i.txt'); C = textscan(fileID, '%f'); PA2_output_i = C{1, 1}; fclose(fileID);
fileID = fopen('Data_valid\PA2_output_q.txt'); C = textscan(fileID, '%f'); PA2_output_q = C{1, 1}; fclose(fileID);
fileID = fopen('Data_valid\PA3_output_i.txt'); C = textscan(fileID, '%f'); PA3_output_i = C{1, 1}; fclose(fileID);
fileID = fopen('Data_valid\PA3_output_q.txt'); C = textscan(fileID, '%f'); PA3_output_q = C{1, 1}; fclose(fileID);

fileID = fopen('Data_valid\Sys_output_i.txt');C = textscan(fileID, '%f'); Sys_output_i = C{1, 1}; fclose(fileID);
fileID = fopen('Data_valid\Sys_output_q.txt');C = textscan(fileID, '%f'); Sys_output_q = C{1, 1}; fclose(fileID);
%% 数据处理
dim = length(Sys_input_i);
% 这么写的原因是因为其奇数位的数据为时间
[Sys_x_i, Sys_x_q,PA1_y_i, PA1_y_q,PA2_y_i, PA2_y_q,PA3_y_i, PA3_y_q,Sys_y_i, Sys_y_q] = deal(zeros(dim, 1));
for k = 1:dim
    [Sys_x_i(k), Sys_x_q(k), Sys_y_i(k), Sys_y_q(k)] = deal(Sys_input_i( k), Sys_input_q( k), Sys_output_i( k), Sys_output_q( k));
    [PA1_y_i(k), PA1_y_q(k)]=deal(PA1_output_i( k),PA1_output_q( k));
    [PA2_y_i(k), PA2_y_q(k)]=deal(PA2_output_i( k),PA2_output_q( k));
    [PA3_y_i(k), PA3_y_q(k)]=deal(PA3_output_i( k),PA3_output_q( k));
    
end 

Data_matrix=[Sys_x_i, Sys_x_q,...
             PA1_y_i, PA1_y_q,...
             PA2_y_i, PA2_y_q,...
             PA3_y_i, PA3_y_q,...
             Sys_y_i, Sys_y_q];
end
