function nnet()
    [trainX,trainT] = importd('bivariate','train');
    neuralnet(trainX,trainT);
    
end

function [] = neuralnet (trainX,trainT)
%     Initializing the parameters
    eta = 1 ;
    alpha = 0.2;
    tol = 0.001;
     nodesPerLayer = [size(trainX,2) 5 1];
%     trainX = [0.1 0.1; 0.1 0.95; 0.95 0.1;0.95 0.95];
%     trainT = [0.1 0.95 0.95 0.1]';
%     nodesPerLayer = [size(trainX,2) 5 1];
    activationFn = {'linear' ; 'logistic'; 'linear'};
    batchmode(nodesPerLayer,trainX,trainT,eta,alpha,tol,activationFn);
    
end

function [error] = batchmode(nodesPerLayer,trainX,trainT,eta,alpha,tol,activationFN)
%     Initializing the weight and gradient parameters
    [weight,deltaWeight,gradient] = init(nodesPerLayer);
    sumerror = 2*tol;
    gradChange = 2*tol;
    numSamples = size(trainX,1);
    trainfunc = cell(size(activationFN));
    trainfuncD = cell(size(activationFN));
    for index =1 : length(activationFN)
        trainfunc {index} = str2func(activationFN{index});
        trainfuncD {index} = str2func([activationFN{index} 'D']);
    end
    epoch = 1;
    hold on;
    while(sumerror > tol && gradChange > tol)
        sumerror = 0;
        ERR = zeros(size(trainT(1)));
        for k = 1:numSamples
           [sactivationOutputs] = forwardPass(nodesPerLayer,trainX(k,:),weight,trainfunc);
           ERR = ERR + [1  trainT(k)] - sactivationOutputs{end};
           sumerror = sumerror+sum(ERR.^2);
        end
        sumerror = sumerror/numSamples;
        ERR=ERR./numSamples;
        [weight,deltaWeight,gradient,gradChange]=backwardPass(nodesPerLayer,ERR',sactivationOutputs,weight,deltaWeight,gradient,trainfuncD,eta,alpha);
        display(['epoch ' num2str(epoch) ' : ' num2str(sumerror)  ' - gradChange ' num2str(gradChange)]);
        plot(epoch,sumerror,'r+');
        plot(epoch,gradChange,'bo');
        pause(0.01);
        epoch=epoch+1;
    end
    hold off;
end

function [error] = patternmode(nodesPerLayer,trainX,trainT,eta,alpha,tol,activationFN)
%     Initializing the weight and gradient parameters
    [weight,deltaWeight,gradient] = init(nodesPerLayer);
    sumerror = 2*tol;
    gradChange = 2*tol;
    numSamples = size(trainX,1);
    trainfunc = cell(size(activationFN));
    trainfuncD = cell(size(activationFN));
    for index =1 : length(activationFN)
        trainfunc {index} = str2func(activationFN{index});
        trainfuncD {index} = str2func([activationFN{index} 'D']);
    end
    
    epoch = 1;
    hold on;
    while(sumerror > tol && gradChange > tol)
        sumerror = 0;
        gradChange = 0;
        for k = 1:numSamples
           [sactivationOutputs] = forwardPass(nodesPerLayer,trainX(k,:),weight,trainfunc);
           ERR = [1  trainT(k)] - sactivationOutputs{end};
           [weight,deltaWeight,gradient,tgradChange]=backwardPass(nodesPerLayer,ERR',sactivationOutputs,weight,deltaWeight,gradient,trainfuncD,eta,alpha);
           gradChange=gradChange+tgradChange;
           sumerror = sumerror+sum(abs(ERR));
        end
        gradChange = gradChange/numSamples;
        sumerror = sumerror/numSamples;
        display(['epoch ' num2str(epoch) ' : ' num2str(sumerror) ' - gradChange ' num2str(gradChange)]);
        plot(epoch,sumerror,'r+');
        plot(epoch,gradChange,'bo');
        pause(0.01);
        epoch=epoch+1;
    end
    hold off;
end

function [sactivationOutputs] = forwardPass(nodesPerLayer,train,weight,func)
    numLayers = length(nodesPerLayer);
    sactivationOutputs  = cell(numLayers,1);
    sactivationOutputs{1} = [1  train];
    % Forward Pass
    for index =2 : numLayers -1
        sactivationOutputs{index} = [1 func{index}(sactivationOutputs{index-1}*weight{index-1})];
    end
     sactivationOutputs{end} = [1 func{end}(sactivationOutputs{end-1}*weight{end})];
end

function [weight,deltaOldWeight,gradientOld,gradChange] = backwardPass(nodesPerLayer,err,sactivationOutputs,weight,deltaOldWeight,gradientOld,trainfuncD,eta,alpha)
    gradChange = 0;
    numWeightComponent = length(nodesPerLayer)-1;
    gradient = cell(numWeightComponent,1); 
    gradient{end} = err.*trainfuncD{end}(sactivationOutputs{end})';
    for index = numWeightComponent:-1:2
        gradient{index-1}=weight{index}*gradient{index}(2:end).*trainfuncD{index}(sactivationOutputs{index})';
    end
    for index = 1:numWeightComponent-1
        deltaWeight=sactivationOutputs{index}'*gradient{index}(2:end)';
        gradChange = gradChange+sum(abs(gradient{index} - gradientOld{index}));
        gradientOld{index} = gradient{index};
        weight{index}=weight{index}+eta*deltaWeight+alpha*deltaOldWeight{index};
        deltaOldWeight{index} = deltaWeight;
    end
    deltaWeight=sactivationOutputs{end-1}'*gradient{end}(2:end)';
    gradChange = gradChange+sum(abs(gradient{end} - gradientOld{end}));
    gradientOld{end} = gradient{end};
    weight{end}=weight{end}+eta*deltaWeight+alpha*deltaOldWeight{end};
    gradChange = gradChange+sum(abs(deltaOldWeight{end} - deltaWeight));
    deltaOldWeight{end} = deltaWeight;
end


function [out] = logistic(in)
        out = 1./(1+exp(-in));
end

function [out] = hTangent(in)
        out = tanh(in);
end

function [out] = linear(in)
    out = in;
end

function [out] = linearD(in)
    out = ones(size(in));
end

function [out] = hTangentD(in)
%     out = (1-hTangent(in).*hTangent(in)); 
    out = (1-in.*in); 
end
function [out] = logisticD(in)
%     out = logistic(in).*(1-logistic(in));
    out = in.*(1-in);
end

function [weight,deltaWeight,gradient] = init(nodesPerLayer)
    numWeightComponents = length(nodesPerLayer)-1;
    weight = cell(numWeightComponents,1);
    deltaWeight = cell(numWeightComponents,1);
    gradient = cell(numWeightComponents,1);
%     Initializing the weight deltaWeight and deltaOldWeight
    for index = 1:numWeightComponents
        weight{index} = 2*rand(nodesPerLayer(index)+1,nodesPerLayer(index+1))-1; 
        deltaWeight{index} = zeros(nodesPerLayer(index)+1,nodesPerLayer(index+1)); 
    end
    for index = 1:numWeightComponents
        gradient{index} = zeros(nodesPerLayer(index+1)+1,1);
    end
end
