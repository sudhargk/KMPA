function nnet()
    [trainX,trainT] = importd('bivariate','train');
    neuralnet(trainX,trainT);
    
end

function [] = neuralnet (trainX,trainT)
%     Initializing the parameters
    eta = 1.0;
    alpha = 0.9;
    tol = 0.001;
%     nodesPerLayer = [size(trainX,2) 5 1];
%     trainX = [0.1 0.1; 0.1 0.95; 0.95 0.1;0.95 0.95];
%     trainT = [0.1 0.95 0.95 0.1]';
    nodesPerLayer = [size(trainX,2) 5 1];
    patternmode(nodesPerLayer,trainX,trainT,eta,alpha,tol,'logistic');
    
end

function [error] = patternmode(nodesPerLayer,trainX,trainT,eta,alpha,tol,activationFN)
%     Initializing the weight parameters
    [weight,deltaWeight] = initWeight(nodesPerLayer);
%     Initializing the S parameters
    sumerror = 2*tol;
    numSamples = size(trainX,1);
    trainfunc = str2func(activationFN);
    trainfuncD = str2func([activationFN 'D']);
    epoch = 1;
    while(sumerror > tol)
        sumerror = 0;
        for k = 1:numSamples
           [sactivationOutputs] = forwardPass(nodesPerLayer,trainX(k,:),weight,trainfunc);
           ERR =  trainT(k) - sactivationOutputs{end};
           [weight,deltaWeight]=backwardPass(nodesPerLayer,ERR,sactivationOutputs,weight,deltaWeight,trainfuncD,eta,alpha);
           sumerror = sumerror+sum(ERR.^2);
        end
        display(['epoch ' num2str(epoch) ' : ' num2str(sumerror)]);
        epoch=epoch+1;
    end
end

% function [error] = patternmode(nodesPerLayer,trainX,trainT,eta,alpha,tol,activationFN)
% %     Initializing the weight parameters
%     [weight,deltaWeight] = initWeight(nodesPerLayer);
% %     Initializing the S parameters
%     [activationOutputs] = initActivationOutputs(nodesPerLayer);
%     sumerror = 2*tol;
%     numSamples = size(trainX,1);
%     trainfunc = str2func(activationFN);
%     trainfuncD = str2func([activationFN 'D']);
%     epoch = 1;
%     while(sumerror > tol)
%         sumerror = 0;
%         for k = 1:numSamples
%            activationOutputs{1} = [1 trainX(k,:)];
%            [activationOutputs,sactivationOutputs] = forwardPass(nodesPerLayer,activationOutputs,weight,trainfunc);
%            ERR =  trainT(k) -sactivationOutputs{end};
%            [weight,deltaWeight]=backwardPass(nodesPerLayer,ERR,sactivationOutputs,weight,deltaWeight,trainfuncD,eta,alpha);
%            sumerror = sumerror+sum(ERR.^2);
%         end
%         display(['epoch ' num2str(epoch) ' : ' num2str(sumerror)]);
%         epoch=epoch+1;
%     end
% end
% function [activationOutputs,sactivationOutputs] = forwardPass(nodesPerLayer,activationOutputs,weight,func)
%     numLayers = length(nodesPerLayer);
%     sactivationOutputs  = cell(size(activationOutputs));
%      sactivationOutputs{1} = activationOutputs{1};
%     % Forward Pass
%     for index =2 : numLayers-1
%         activationOutputs{index} = sactivationOutputs{index-1}*weight{index-1};
%         sactivationOutputs{index} = [1 func(activationOutputs{index})];
%     end
%     activationOutputs{end} = sactivationOutputs{end-1}*weight{end};
%     sactivationOutputs{end} = func(activationOutputs{end});
% end

function [sactivationOutputs] = forwardPass(nodesPerLayer,train,weight,func)
    numLayers = length(nodesPerLayer);
    sactivationOutputs  = cell(numLayers,1);
    sactivationOutputs{1} = [1  train];
    % Forward Pass
    for index =2 : numLayers-1
        sactivationOutputs{index} = [1 func(sactivationOutputs{index-1}*weight{index-1})];
    end
    sactivationOutputs{end} = func(sactivationOutputs{end-1}*weight{end});
end

function [weight,deltaOldWeight] = backwardPass(nodesPerLayer,err,sactivationOutputs,weight,deltaOldWeight,trainfuncD,eta,alpha)
    numWeightComponent = length(nodesPerLayer)-1;
    gradient = cell(numWeightComponent,1); 
    gradient{end} = err.*trainfuncD(sactivationOutputs{end});
    for index = numWeightComponent:-1:2
        gradient{index-1}=weight{index}*gradient{index}.*trainfuncD(sactivationOutputs{index})';
    end
    for index = 1:numWeightComponent-1
        deltaWeight=sactivationOutputs{index}'*gradient{index}(2:end)';
        weight{index}=weight{index}+eta*deltaWeight+alpha*deltaOldWeight{index};
        deltaOldWeight{index} = deltaWeight;
    end
    deltaWeight=sactivationOutputs{end-1}'*gradient{end}';
    weight{end}=weight{end}+eta*deltaWeight+alpha*deltaOldWeight{end};
    deltaOldWeight{end} = deltaWeight;
end


function [out] = logistic(in)
        out = 1./(1+exp(-in));
end

function [out] = hTangent(in)
        out = tanh(in);
end

function [out] = hTangentD(in)
%     out = (1-hTangent(in).*hTangent(in)); 
    out = (1-in.*in); 
end
function [out] = logisticD(in)
%     out = logistic(in).*(1-logistic(in));
    out = in.*(1-in);
end

function [weight,deltaWeight] = initWeight(nodesPerLayer)
    numWeightComponents = length(nodesPerLayer)-1;
    weight = cell(numWeightComponents,1);
    deltaWeight = cell(numWeightComponents,1);
%     Initializing the weight deltaWeight and deltaOldWeight
    for index = 1:numWeightComponents
        weight{index} = 2*rand(nodesPerLayer(index)+1,nodesPerLayer(index+1))-1; 
        deltaWeight{index} = zeros(nodesPerLayer(index)+1,nodesPerLayer(index+1)); 
    end
end

function [activationOutputs] = initActivationOutputs(nodesPerLayer)
    numActivationComponents = length(nodesPerLayer);
    activationOutputs = cell(numActivationComponents,1);
    assert(numActivationComponents>=3);
    activationOutputs{1} = zeros(1,nodesPerLayer(1));
    for index = 2:numActivationComponents-1
       activationOutputs{index} =  zeros(1,nodesPerLayer(index));
    end
    activationOutputs{numActivationComponents}=zeros(1,nodesPerLayer(numActivationComponents));
end