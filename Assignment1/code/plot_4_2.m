%%Plots target output and model output vs input feature vector for
%%bivariate data
% numbasis : number of basis
% variance : variance parameter for the guassian basis
% lambda : regularization parameter
% E.g 
%       plot_4_2(10,exp(-10),0.05);

function [] =plot_4_2(basis,lambda,variance)
    data = 'bivariate';
    [trainX,trainT] = importd(data,'train');
    [testX,testT] = importd(data,'test');
    [valX,valT] = importd(data,'val');
    
    [trainX,testX,valX] = normalize(trainX,testX,valX);
    
    [M,~] = computeKMeans(trainX, basis);
    trainXPhi = computeDesignMatrix(trainX,'Gaussian',basis,variance,M);
    testXPhi = computeDesignMatrix(testX,'Gaussian',basis,variance,M);
    valXPhi = computeDesignMatrix(valX,'Gaussian',basis,variance,M);
    
    W = train(trainXPhi,trainT,lambda);
    
    numPoints = 100;
    ticks = linspace(0,1,numPoints); % Range is [0,1] on both axes due to normalization
    [X1, X2] = meshgrid(ticks, ticks);
    plotXPhi = computeDesignMatrix([X1(:) X2(:)],'Gaussian',basis,variance,M);
    Y = reshape(plotXPhi*W, numPoints, numPoints);
    
    generatePlot(trainX,trainXPhi*W,trainT,X1,X2,Y,'train');
    generatePlot(testX,testXPhi*W,testT,X1,X2,Y,'test');
    generatePlot(valX,valXPhi*W,valT,X1,X2,Y,'validation');
    
end

function [] = generatePlot(input, output, target, modelX1, modelX2, modelY, datatype)
    figure();
    set(gcf, 'WindowStyle', 'docked');
    
    hold on;
    surf(modelX1, modelX2, modelY);
    X1 = input(:,1);
    X2 = input(:,2);
    Y = output(:,1);
    T = target(:,1);
    plot3(X1,X2,T,'g+',X1,X2,Y,'ro');
    title(['Plot of Target output and Model output on ' datatype ' data for bivariate dataset']);
    legend('Approximated function','Target Output','Model Output');
    xlabel('Dimension 1');
    ylabel('Dimension 2');
    zlabel('Output');
    view([-22 23]);
    hold off;
end
