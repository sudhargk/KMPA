
%%
%
%%
function plot_4()
    mode = 'batch';
    dataset = 'bivariate';
    [trainX, trainT] = importd(dataset, 'train');
    [valX, valT] = importd(dataset, 'val');
    [testX, testT] = importd(dataset, 'test');
    inputs = [trainX valX testX];
    targets = [trainT valT testT];
    
    if (strcmp(mode, 'pattern')==1)
        inputs = mat2cell(inputs, size(inputs,1), ones(1, size(inputs,2)));
        targets = mat2cell(targets, size(targets,1), ones(1, size(targets,2)));
    end
    trainingMethod = 'traingdm'; % scaled conjugate gradient
    
    numDim = size(trainX,1);
    numClasses = size(trainT,1);
    
    % Set initialization parameters
    eta = 0.05;
    alpha = 0.9;
    tol = 1e-3;
    max_epochs = 10000; % set no of epochs to be very large
    global BETA;
    BETA = 0.3;
    nodesPerLayer = [numDim 30 numClasses];
    activationFcns = {'tansig','purelin'};
    initializationFcn = 'rands';
    
    setdemorandstream(exp(1));
    
    %% Should not require too many changes often
    % Create a Pattern Recognition Network
    net = fitnet;
    
    % Choose Input and Output Pre/Post-Processing Functions
    % For a list of all processing functions type: help nnprocess
    net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
    net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};
    
    % Layer configuration
    for i = 1 : net.numLayers
        net.layers{i}.size = nodesPerLayer(i+1); % set no of hidden layer nodes
        net.layers{i}.transferFcn = activationFcns{i}; % set activation fn for each layer
    end
    
    % Initialization of weights
    net.initFcn = 'initlay';
    for i = 1 : net.numLayers
        %     net.layers{i}.initFcn = 'initnw';
        net.layers{i}.initFcn = 'initwb';
        net.biases{i}.initFcn = initializationFcn;
        for j = 1 : net.numLayers
            net.layerWeights{i,j}.initFcn = initializationFcn;
        end
        for j = 1 : net.numInputs
            net.inputWeights{i,j}.initFcn = initializationFcn;
        end
    end
    
    % Setup Division of Data for Training, Validation, Testing
    % For a list of all data division functions type: help nndivide
    net.divideFcn = 'divideind';  % Divide data as per given indices
    b = 1; e = size(trainX,2);
    net.divideParam.trainInd = b:e;
    b = e + 1; e = e + size(valX,2);
    net.divideParam.valInd = b:e;
    b = e + 1; e = e + size(testX,2);
    net.divideParam.testInd = b:e;
    
    % Choose training parameters
    switch (mode)        
        case 'batch'
             switch (trainingMethod)
                case 'traingdm'
                    net.trainFcn = 'traingdm';  % Gradient descent w/ momentum
                    net.trainParam.lr = eta;
                    net.trainParam.mc = alpha;
                case 'trainscg'
                    net.trainFcn = 'trainscg';  % Scaled conjugate gradient
                    net.trainParam.sigma = 1e-5;
                    net.trainParam.lambda = 1e-7;
                otherwise
                    fprintf(stderr, 'Training method can only be conjugate gradient or gradient descent with momentum');
             end
            
            % termination criteria
            net.trainParam.time = Inf; % time limit
            net.trainParam.goal = tol;
            net.trainParam.max_fail = 1000; % max validation failures
            %(no of consecutive epochs validation error fails to decrease)
            net.trainParam.min_grad = tol; % norm of error gradient
            
        otherwise
            fprintf(stderr, 'Mode can only be pattern or batch');
    end
    
    % Set parameters for gradient descent w/ momentum
    for i = 1 : net.numLayers
        if (strcmp(net.biases{i}.learnFcn, 'learngdm') == 1)
            net.biases{i}.learnParam.lr = eta;
            net.biases{i}.learnParam.mc = alpha;
        end
        for j = 1 : net.numLayers
            if (strcmp(net.biases{i}.learnFcn, 'learngdm') == 1)
                net.layerWeights{i,j}.learnParam.lr = eta;
                net.layerWeights{i,j}.learnParam.mc = alpha;
            end
        end
        for j = 1 : net.numInputs
            if (strcmp(net.biases{i}.learnFcn, 'learngdm') == 1)
                net.inputWeights{i,j}.learnParam.lr = eta;
                net.inputWeights{i,j}.learnParam.mc = alpha;
            end
        end
    end
    
    % display parameters
    net.trainParam.show = 10;
    net.trainParam.showCommandLine = 0;
    net.trainParam.showWindow = 1;
    
    
    % Choose a Performance Function
    % For a list of all performance functions type: help nnperformance
    net.performFcn = 'mse';  % Mean squared error
    
    % Choose Plot Functions
    % For a list of all plot functions type: help nnplot
    net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
        'plotconfusion', 'plotroc'};
    
%     for epoch = [5 10]
         net.trainParam.epochs = max_epochs;
%         Train the Network
         [net,tr] = train(net,inputs,targets);
%         D = inputs';
%         mn = min(D);
%         mx = max(D);
%         n = 500; % grid size
%         [gridX, gridY] = meshgrid( linspace(mn(1),mx(1),n), linspace(mn(2),mx(2),n) );
%         Xl = gridX(:)'; Yl = gridY(:)';
%         gridSout = net([Xl; Yl]);
%         gridSout = reshape(gridSout', [n n numClasses]);
%         
%         inc = 5;
%         gridXS = gridX(1:inc:end, 1:inc:end);
%         gridYS = gridY(1:inc:end, 1:inc:end);
%         gridSoutS = gridSout(1:inc:end, 1:inc:end,:);
%         
%         plotHiddenOutputSurface(net, gridXS, gridYS,1,epoch);
%          plotOutputSurface(gridXS,gridYS,gridSoutS,epoch);
%         
%     end
%     % Test the Network
%     outputs = net(inputs);
   
end

function  plotOutputSurface(modelX1, modelX2, modelY,epoch)
    figure(),set(gcf, 'WindowStyle', 'docked');
    hold on;
    surf(modelX1, modelX2, modelY, 'FaceColor', [0.5 0.4 0.4]);
    hold off;
    title(['Output surface  for bivariate dataset at epoch ' num2str(epoch)]);
end

function plotHiddenOutputSurface(net, gridX, gridY, layerNo,epoch)
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
    for j = 2:5:J
        figure(), set(gcf, 'WindowStyle', 'docked'),
        xlabel('Dimension 1'), ylabel('Dimension 2'), zlabel('Hidden layer output')
        surf(gridX, gridY, s(:,:,j),'FaceColor',colors(j,:));
        axis tight;
        title(['Output surfaces of node ' num2str(j) ' in hidden layer ' num2str(layerNo) 'after training at epoch ' num2str(epoch)]),
    end
end