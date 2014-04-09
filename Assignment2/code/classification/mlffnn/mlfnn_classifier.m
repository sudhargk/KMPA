function mlfnn_classifier()
    
    dataset = 'nonlinearlySeparable';
    [trainX, trainT] = importd(dataset, 'train');
    [valX, valT] = importd(dataset, 'val');
    [testX, testT] = importd(dataset, 'test');
    inputs = [trainX valX testX];
    targets = [trainT valT testT];
    
    numDim = size(trainX,1);
    numClasses = size(trainT,1);
    
    % Set initialization parameters
    mode = 'batch';
    trainingMethod = 'traingdm'; % gradient descent with momentum
%     trainingMethod = 'trainscg'; % scaled conjugate gradient
    eta = 20;
    alpha = 0.5;
    gradtol = 1e-5;
    errtol = 1e-5;
    max_epochs = 100000; % set no of epochs to be very large
    validation_checks = 10;
    nodesPerLayer = [numDim 10 numClasses];
    activationFcns = {'tansig', 'logsig'};
    initializationFcn = 'rands';
    
    setdemorandstream(pi); % random seeding for reproducible results
    
    %% Should not require too many changes often
    % Create a Pattern Recognition Network
    net = patternnet;
    
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
        case 'pattern'
            net.trainFcn = 'trainr';  % Random sequencing of inputs for pattern mode
            % termination criteria
            net.trainParam.epochs = max_epochs;
            net.trainParam.time = Inf; % time limit
            net.trainParam.goal = errtol;
            net.trainParam.min_grad = gradtol;
            
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
            net.trainParam.epochs = max_epochs;
            net.trainParam.time = Inf; % time limit
            net.trainParam.goal = errtol;
            net.trainParam.max_fail = validation_checks; % max validation failures
            %(no of consecutive epochs validation error fails to decrease)
            net.trainParam.min_grad = gradtol; % norm of error gradient
            
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
    
    % Display parameters
    net.trainParam.show = 10;
    net.trainParam.showCommandLine = 0;
    net.trainParam.showWindow = 1;
    
    
    % Choose a Performance Function
    % For a list of all performance functions type: help nnperformance
    net.performFcn = 'mse';  % Mean squared error
    
    % Choose Plot Functions
    % For a list of all plot functions type: help nnplot
    net.plotFcns = {'plotperform','plottrainstate', ...
        'plotconfusion', 'plotroc'};
    
    
    % Train the Network
    net = configure(net, inputs, targets);
%     load('patmodedata');
    [net,tr] = train(net,inputs,targets);
%     bias = net.b; inputwts = net.iw; layerwts = net.lw;
%     save('patmodedata', 'bias', 'inputwts', 'layerwts');
    
    % Test the Network
    outputs = net(inputs);
    errors = gsubtract(targets,outputs);
    performance = perform(net,targets,outputs)
    
    % Recalculate Training, Validation and Test Performance
    trainTargets = targets .* tr.trainMask{1};
    valTargets = targets  .* tr.valMask{1};
    testTargets = targets  .* tr.testMask{1};
    trainPerformance = perform(net,trainTargets,outputs)
    valPerformance = perform(net,valTargets,outputs)
    testPerformance = perform(net,testTargets,outputs)
    
    testOut = net(trainX);
    % View the Network
    % view(net)
    
    D = inputs';
    mn = min(D);
    mx = max(D);
    n = 500; % grid size
    [gridX, gridY] = meshgrid( linspace(mn(1),mx(1),n), linspace(mn(2),mx(2),n) );
    Xl = gridX(:)'; Yl = gridY(:)';
    gridSout = net([Xl; Yl]);
    gridSout = reshape(gridSout', [n n numClasses]);
    
    inc = 20;
    gridXS = gridX(1:inc:end, 1:inc:end);
    gridYS = gridY(1:inc:end, 1:inc:end);
    gridSoutS = gridSout(1:inc:end, 1:inc:end,:);
    
    % PLOTS
    % Uncomment these lines to enable various plots.
    %figure, plottrainstate(tr)
    %figure, ploterrhist(errors)
    
%     figure, set(gcf, 'WindowStyle', 'docked'), plotconfusion(testT,testOut)
%     figure, set(gcf, 'WindowStyle', 'docked'), plotroc(targets,outputs)
%     figure, set(gcf, 'WindowStyle', 'docked'), plotperform(tr)
    figure, set(gcf, 'WindowStyle', 'docked'), plotdecisionregions(testX, testT, testOut, gridX, gridY, gridSout)
    figure, set(gcf, 'WindowStyle', 'docked'), plotoutputs(gridXS, gridYS, gridSoutS)
%     plothiddenlayeroutputs(net, gridXS, gridYS, 1)
   
end

