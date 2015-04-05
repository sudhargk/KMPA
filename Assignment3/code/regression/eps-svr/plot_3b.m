%%Plots target output and model output vs input feature vector for
%%bivariate data
% numbasis : number of basis
% variance : variance parameter for the guassian basis
% lambda : regularization parameter
% E.g 
%       plot_3b(10,9,0);

function [] = plot_3b(cost,gamma,coef)
    
%    kernel =  'gaussian';
    if(nargin<3)
        
        coef = 0; 
        if(nargin<4) 
            gamma = 3; 
        end
        cost = 1; 
    end
    dataset = 'bivariate';
    eps=0.1;
    [trainX,trainT] = importd(dataset,'train');
    [testX,testT] = importd(dataset,'test');
    [valX,valT] = importd(dataset,'val');
    
    [trainX,testX,valX] = normalize(trainX,testX,valX);
%    [svroptions] = buildsvroptions(cost,kernel,a,b,d);
    [svroptions] = buildsvroptions(cost,coef,gamma,eps);
    [model]=c_svr_train(trainX,svroptions,trainT);
    trainET = c_svr_test(trainX,model);
    valET = c_svr_test(valX,model);
    testET = c_svr_test(testX,model);

    numPoints = 100;
    ticks = linspace(0,1,numPoints); % Range is [0,1] on both axes due to normalization
    [X1, X2] = meshgrid(ticks, ticks);
    Y = c_svr_test([X1(:) X2(:)],model);
    Y = reshape(Y, numPoints, numPoints);
  
    generatePlot(trainX,trainET,trainT,X1,X2,Y,'train','b+');
    generatePlot(testX,testET,testT,X1,X2,Y,'test','m+');
    generatePlot(valX,valET,valT,X1,X2,Y,'validation','y+');
    
end

function [] = generatePlot(input, output, target, modelX1, modelX2, modelY, datatype, color)
    figure();
    set(gcf, 'WindowStyle', 'docked');
    hold on;
    surf(modelX1, modelX2, modelY, 'FaceColor', [0.5 0.4 0.4]);
    X1 = input(:,1);
    X2 = input(:,2);
    Y = output(:,1);
    T = target(:,1);
    plot3(X1,X2,T,color);
    hold off;
    title(['Plot of Target output and Model output on ' datatype ' data for bivariate dataset']);
    legend('Approximated function','Target Output');
    xlabel('Dimension 1');
    ylabel('Dimension 2');
    zlabel('Output');
    view([-22 23]);
    
end
