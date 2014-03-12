%%Plots target output and model output vs input feature vector for
%%univariate data
% numbasis : number of basis
% lambda : regularization parameter
% E.g 
%       plot_2u(10,exp(-10));

function [] =plot_2u(basis,lambda)
    data = 'univariate';
    basisType='Gaussian';
    [trainX,trainT] = importd(data,'train');
    numPoints = 1000;
    step = length(trainT) / numPoints;
    idx = round((1:numPoints) * step);
    trainX = trainX(idx);
    trainT = trainT(idx);
    [testX,testT] = importd(data,'test');
    [valX,valT] = importd(data,'val');
    
    [trainX,testX,valX] = normalize(trainX,testX,valX);
    [M,tichonovDist,width] = computeClusterMeans(trainX,basis);
    [trainXPhi] = computeDesignMatrix(trainX,basisType,basis,M,width);
    testXPhi = computeDesignMatrix(testX,basisType,basis,M,width);
    valXPhi = computeDesignMatrix(valX,basisType,basis,M,width);
    W = train(trainXPhi,trainT,lambda,tichonovDist);
    
    generatePlot(trainX,trainXPhi*W,trainT,'train');
    generatePlot(testX,testXPhi*W,testT,'test');
    generatePlot(valX,valXPhi*W,valT,'validation');
    
end

function [] = generatePlot(input, output, target, dataType)
    figure();
    set(gcf, 'WindowStyle', 'docked');
    
    X = input(:,1);
    Y = output(:,1);
    T = target(:,1);
    
    plot(X, T, 'go', X, Y,'r');
    legend('Target Output','Model Output');
    xlabel('Input');
    ylabel('Output');
    title(['Plot of Target output and Model output on ' dataType ' data for univariate dataset']);
    
end
