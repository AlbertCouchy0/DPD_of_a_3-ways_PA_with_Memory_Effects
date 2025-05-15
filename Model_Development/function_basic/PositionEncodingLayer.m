%% 自定义位置编码层（添加到网络前需定义此类）
classdef PositionEncodingLayer < nnet.layer.Layer
    properties (Learnable)
        PositionEncoding
    end
    
    methods
        function layer = PositionEncodingLayer(seqLen, d_model, name)
            layer.Name = name;
            layer.PositionEncoding = randn(d_model, seqLen); % 可学习的位置编码
        end
        
        function Z = predict(layer, X)
            Z = X + layer.PositionEncoding;
        end
    end
end