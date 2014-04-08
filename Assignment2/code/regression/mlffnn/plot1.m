
%%
%
%%
function plot1()
    mode = 'batch';
    dataset = 'multivariate';
    [trainX, trainT] = importd(dataset, 'train');
    [valX, valT] = importd(dataset, 'val');
    [testX, testT] = importd(dataset, 'test');
     trainingMethod = 'traingdm'; % scaled conjugate gradient

    inputs = [trainX valX testX];
    targets = [trainT valT testT];
     
    numDim = size(trainX,1);
    numClasses = size(trainT,1);
    
    
    
    % Set initialization parameters
    eta = 0.01;
    alpha = 0.9;
    tol = 1e-3;
    max_epochs = 10000; % set no of epochs to be very large
    global BETA;
    BETA = 1;
    modelComplexity = [1 3 5 10 20 30 40 50 60 70 80 90 100 125 150 175 200]
    index=1;
    
    trainPerformance = zeros(1,length(modelComplexity));
    valPerformance = zeros(1,length(modelComplexity));
    testPerformance = zeros(1,length(modelComplexity));
   
    for complexity = modelComplexity
    nodesPerLayer = [numDim complexity numClasses];
    activationFcns = {'tansig','purelin'};
    initializationFcn = 'rands';
    setdemorandstream(exp(1));
    
    %% Should not require too many changes often
    % Create a Pattern Recognition Network
    net = fitnet;
    
    
    net.numLayers = length(nodesPerLayer)-1; % 1 output layer counted here
  
    
    % Choose Input and Output Pre/Post-Processing Functions
    % For a list of all processing functions type: help nnprocess
    net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
    net.outputs{net.numLayers}.processFcns = {'removeconstantrows','mapminmax'};
    
    % Layer configuration
    for i = 1 : net.numLayers
        net.layers{i}.size = nodesPerLayer(i+1); % set no of hidden layer nodes
        net.layers{i}.transferFcn = activationFcns{i}; % set activation fn for each layer
    end
    
    % Initialization of weights
     net.initFcn = 'initlay';
    switch (initializationFcn)
        case 'rands'
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
        case 'initnw'
            for i = 1 : net.numLayers
                %     net.layers{i}.initFcn = 'initnw';
                net.layers{i}.initFcn = initializationFcn;
                net.biases{i}.initFcn = initializationFcn;
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
            net.trainFcn = 'trains';  % Random sequencing of inputs for pattern mode
            % termination criteria
            net.trainParam.epochs = max_epochs;
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
    
    
    
    % Train the Network
    [net,tr] = train(net,inputs,targets);
    
    % Test the Network
     outputs = net(inputs);
%     errors = gsubtract(targets,outputs);
%     performance = perform(net,targets,outputs)
    
    % Recalculate Training, Validation and Test Performance
    trainTargets = targets .* tr.trainMask{1};
    valTargets = targets  .* tr.valMask{1};
    testTargets = targets  .* tr.testMask{1};
    trainPerformance(index) = perform(net,trainTargets,outputs);
    valPerformance(index) = perform(net,valTargets,outputs);
    testPerformance(index) = perform(net,testTargets,outputs);
    index=index+1;
    end
    
    % PLOTS
    figure,
    plot(modelComplexity, trainPerformance, 'b',...
        modelComplexity,valPerformance, 'r',...
        modelComplexity,  testPerformance, 'm');
    title({'mean squared error on train, test and validation data'
        'as functions of the number of basis functions'});
    ylabel('E_{RMS}');
    xlabel('Number of basis functions');
    legend('Train error', 'Validation error', 'Test error');
end

