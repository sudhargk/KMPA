function net = initNet(net, initializationFcn)
    % initializationFcn = 'rands';
    
    net.initFcn = 'initlay';
    for i = 1 : net.numLayers
        net.layers{i}.initFcn = 'initwb';
    end
    for i = 1 : net.numLayers
        for j = 1 : net.numLayers
            net.layerWeights{i,j}.initFcn = initializationFcn;
        end
        for j = 1 : net.numInputs
            net.inputWeights{i,j}.initFcn = initializationFcn;
        end
        net.biases{i}.initFcn = initializationFcn;
    end
end
