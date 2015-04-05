function mlfnn_classifier_image()
    
    dataset = 'image';
    [trainX, trainT, classes] = importd(dataset, 'train');
    [valX, valT] = importd(dataset, 'val');
    [testX, testT] = importd(dataset, 'test');
    inputs = [trainX valX testX];
    targets = [trainT valT testT];
    
%     if (strcmp(mode, 'pattern')==1)
%         inputs = mat2cell(inputs, size(inputs,1), ones(1, size(inputs,2)));
%         targets = mat2cell(targets, size(targets,1), ones(1, size(targets,2)));
%     end
    
    numDim = size(trainX,1);
    numClasses = size(trainT,1);
    
    % Set initialization parameters
    mode = 'batch';
    trainingMethod = 'traingdm'; % gradient descent with momentum
%     trainingMethod = 'trainscg'; % scaled conjugate gradient
    eta = 0.3;
    alpha = 0.92;
    gradtol = 0.001;
    errtol = 0.055;
    max_epochs = 100000; % set no of epochs to be very large
    val_checks = 10000;  % max validation failures
            %(no of consecutive epochs validation error fails to decrease)
    global BETA;
    BETA = 1;
    nodesPerLayer = [numDim 100 30 numClasses];
    activationFcns = {'tansig', 'tansig', 'logsig'};
    initializationFcn = 'rands';
    
    setdemorandstream(pi); % random seeding for reproducible results
    
    %% Should not require too many changes often
    % Create a Pattern Recognition Network
    net = patternnet;
    net.numInputs = 1;
    net.numLayers = length(nodesPerLayer)-1; % 1 output layer counted here
    
    % Configure layer connections
    net.biasConnect = ones(net.numLayers, 1); % hidden layers and output layer both have bias connections
    net.inputConnect = eye(net.numLayers, 1); % inputConnect(i,j) is 1 when input j is connected to layer i
    net.layerConnect = diag(ones(net.numLayers-1,1),-1); % layerConnect(i,j) is 1 when layer j's output is input to layer i
    net.outputConnect = zeros(1,net.numLayers); net.outputConnect(net.numLayers) = 1;
    % outputConnect(j) is 1 when layer j gives an output
    % Note that this implicitly determines numOutputs
    
    % Layer configuration
    for i = 1 : net.numLayers
        net.layers{i}.size = nodesPerLayer(i+1); % set no of hidden layer nodes
        net.layers{i}.transferFcn = activationFcns{i}; % set activation fn for each layer
    end
    
    % Choose Input and Output Pre/Post-Processing Functions
    % For a list of all processing functions type: help nnprocess
    net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
    net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};
    
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
            net  .trainParam.epochs = max_epochs;
            net.trainParam.time = Inf; % time limit
            net.trainParam.goal = tol;
            
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
            net.trainParam.max_fail = val_checks;
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
    
    % display parameters
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
    [net,tr] = train(net,inputs,targets);
    
    % Test the Network
    outputs = net(inputs);
    testOut = net(testX);
    errors = gsubtract(targets,outputs);
    performance = perform(net,targets,outputs)
    
    % Recalculate Training, Validation and Test Performance
    trainTargets = targets .* tr.trainMask{1};
    valTargets = targets  .* tr.valMask{1};
    testTargets = targets  .* tr.testMask{1};
    trainPerformance = perform(net,trainTargets,outputs)
    valPerformance = perform(net,valTargets,outputs)
    testPerformance = perform(net,testTargets,outputs)
    
    % View the Network
    % view(net)
    
    % PLOTS
    % Uncomment these lines to enable various plots.
    %figure, plottrainstate(tr)
    
%     figure, set(gcf, 'WindowStyle', 'docked'), plotconfusion(testT,testOut)
    figure, set(gcf, 'WindowStyle', 'docked'), plotroc(testT,testOut), [~, ~, ph] = legend(gca); legend(ph,classes);
%     figure, set(gcf, 'WindowStyle', 'docked'), plotperform(tr)
    
end
