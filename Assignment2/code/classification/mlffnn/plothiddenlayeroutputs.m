function plothiddenlayeroutputs(net, gridX, gridY, layerNo)
    %PLOTHIDDENLAYEROUTPUTS Partially runs the network on given inputs to
    %determine intermediate layer outputs
    
    Xl = gridX(:)'; Yl = gridY(:)';
    n1 = size(gridX,1);
    n2 = size(gridX,2);
    inp = [Xl; Yl];
    inputs = inp;
    for fIndex = 1 : numel(net.inputs{1}.processFcns)
        procFcn = str2func(net.inputs{1}.processFcns{fIndex});
        procFcnSettings = net.inputs{1}.processSettings{fIndex};
        inputs = procFcn('apply', inputs, procFcnSettings);
    end
    
    a1 = bsxfun(@plus, net.iw{1}*inputs, net.b{1});
    actFcn1 = str2func(net.layers{1}.transferFcn);
    actFcnParams1 = net.layers{1}.transferParam;
    s1 = actFcn1(a1,actFcnParams1);
    
    s = s1;
    if (layerNo > 1)
        for lIndex = 2 : layerNo
            a = net.lw{lIndex, lIndex-1}*s;
            if (net.biasConnect(lIndex))
                a = bsxfun(@plus, a, net.b{lIndex});
            end
            actFcn = str2func(net.layers{lIndex}.transferFcn);
            actFncParams = net.layers{lIndex}.transferParam;
            s = actFcn(a, actFncParams);
        end
    end
    
    J = size(s,1);
    s = reshape(s', [n1 n2 J]);
    colors = lines();
    for j = 1:J
        figure, set(gcf, 'WindowStyle', 'docked'),
        xlabel('Dimension 1'), ylabel('Dimension 2'), zlabel('Hidden layer output')
        surf(gridX, gridY, s(:,:,j),'FaceColor',colors(j,:));
        axis tight;
        title(['Output surfaces of node ' num2str(j) ' in hidden layer ' num2str(layerNo) 'after training']),
    end
    
%     %Test code. Check if output is the same as directly calling net
%     ao = bsxfun(@plus, net.lw{2,1}*s1, net.b{2});
%     actFcno = str2func(net.layers{2}.transferFcn);
%     actFcnParamso = net.layers{2}.transferParam;
%     so = actFcno(ao, actFcnParamso);
%     outputs = so;
%     for fIndex = numel(net.outputs{2}.processFcns) : -1 : 1
%         procFcn = str2func(net.outputs{2}.processFcns{fIndex});
%         procFcnSettings = net.outputs{2}.processSettings{fIndex};
%         outputs = procFcn('reverse', outputs, procFcnSettings);
%     end
%     yo = net(inp);
%     assert(all(all(yo-outputs < 1e-15)));
    
end