%%Plots target output and model output vs input feature vector for univariate data
%
% E.g 
%       plot_2u(10,exp(-10));

function [] =plot_3u(dataset,kernel,cost,a,b,d)
   if(nargin<6)
        dataset = 'univariate';
        kernel =  'gaussian';
        cost = 1; a = 0.3; b = 3; d = 2; 
    end
    [trainX,trainT] = importd(dataset,'train');
    [testX,testT] = importd(dataset,'test');
    [valX,valT] = importd(dataset,'val');
    
    [trainX,testX,valX] = normalize(trainX,testX,valX);
    [svroptions] = buildsvroptions(cost,kernel,a,b,d);
    [model]=c_svr_train(trainX,svroptions,trainT);
    trainET = c_svr_test(trainX,model);
    valET = c_svr_test(valX,model);
    testET = c_svr_test(testX,model);

   
    generatePlot(trainX,trainET,trainT,'train');
    generatePlot(testX,valET,testT,'test');
    generatePlot(valX,valET,valT,'validation');
    
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
