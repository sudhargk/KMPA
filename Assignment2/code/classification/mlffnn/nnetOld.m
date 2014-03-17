function nnetOld()
    %Old version -- use mlfnn_classifier() instead
    
    [trainX, trainT] = importd('nonlinearlySeparable', 'train');
    [valX, valT] = importd('nonlinearlySeparable', 'val');
    [testX, testT] = importd('nonlinearlySeparable', 'test');
    inputs = [trainX valX testX];
    targets = [trainT valT testT];
    
    numDim = size(trainX,1);
    numClasses = size(trainT,1);
    
    eta = 0.05;
    alpha = 0.9;
    tol = 1e-3;
    global BETA;
    BETA = 1;
    nodesPerLayer = [numDim 7 numClasses];
    activationFcns = {'tansig', 'logsig'};
    initializationFcn = 'rands';
    
    % Create a Pattern Recognition Network
    % hiddenLayerSize = 10;
    % net = patternnet(hiddenLayerSize);
    
    % Setup network architecture
    
    % Basic configuration
    net = patternnet;
    net.numInputs = 1; % only one set of inputs (this is not input vector dimensionality)
    net.numLayers = length(nodesPerLayer)-1; % 1 output layer counted here
    
    % Configure layer connections
    net.biasConnect = [1; 1]; % hidden layer and output layer both have bias connections
    net.inputConnect = [1; 0]; % inputConnect(i,j) is 1 when input j is connected to layer i
    net.layerConnect = [0 0; 1 0]; % layerConnect(i,j) is 1 when layer j's output is input to layer i
    net.outputConnect = [0 1]; % outputConnect(j) is 1 when layer j gives an output
    % Note that this implicitly determines numOutputs
    net.inputs{1}.exampleInput = trainX; % automatically determines input range, dimensionality etc.
    net.outputs{2}.exampleOutput = trainT; % automatically determines input range, dimensionality etc.
    
    % Choose Input and Output Pre/Post-Processing Functions
    net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'}; % removeconstant rows removes non-discriminatory features
    net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'}; % mapminmax does normalization in [-1,1]
    
    % Setup Division of Data for Training, Validation, Testing
    % For a list of all data division functions type: help nndivide
    net.divideFcn = 'divideind';  % Divide data as per given indices
    b = 1; e = size(trainX,2);
    net.divideParam.trainInd = b:e;
    b = e + 1; e = e + size(valX,2);
    net.divideParam.valInd = b:e;
    b = e + 1; e = e + size(testX,2);
    net.divideParam.testInd = b:e;
    % net.divideFcn = 'dividerand';  % Divide data randomly
    % net.divideMode = 'sample';  % Divide up every sample
    % net.divideParam.trainRatio = 70/100;
    % net.divideParam.valRatio = 15/100;
    % net.divideParam.testRatio = 15/100;
    
    % For help on training function 'trainscg' type: help trainscg
    % For a list of all training functions type: help nntrain
    % net.trainFcn = 'trainscg';  % Scaled conjugate gradient
    net.trainFcn = 'trainscg';
    % net.trainParam.lr = eta; % learning rate
    % net.trainParam.mc = alpha; % momentum factor
    
    % termination criteria
    net.trainParam.epochs = 100000; % set no of epochs to be very large
    net.trainParam.time = Inf; % time limit
    net.trainParam.max_fail = 6; % max validation failures
    %(no of consecutive epochs validation error fails to decrease)
    net.trainParam.min_grad = tol; % norm
    
    % display parameters
    net.trainParam.show = 10;
    net.trainParam.showCommandLine = 0;
    net.trainParam.showWindow = 1;
    
    % Choose a Performance Function
    % For a list of all performance functions type: help nnperformance
    net.performFcn = 'mse';  % Mean squared error
    net.performParam.regularization = 0; % Regularization parameter
    
    % Choose Plot Functions
    % For a list of all plot functions type: help nnplot
    net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
        'plotconfusion', 'plotroc'};
    
    % Initialize biases and weights
    net = init(net);
    % Train the Network
    [net,tr] = train(net,inputs,targets);
    
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
    
    % View the Network
    view(net)
    
    % Plots
    % Uncomment these lines to enable various plots.
    %figure, plotperform(tr)
    %figure, plottrainstate(tr)
    %figure, plotconfusion(targets,outputs)
    %figure, plotroc(targets,outputs)
    %figure, ploterrhist(errors)
