%% Plot of RMS error on train, test and validation data
%  for different model complexity and different regularization parameters
%
function [] = plot_2(dataset,kernel,cost,a,Epsilon)
     if(nargin<6)
        dataset = 'bivariate';
        kernel =  'gaussian';
        cost = 1; a = 3; b = 3; d = 2; 
     end
     
    [trainX,trainT] = importd(dataset,'train');
    [valX,valT] = importd(dataset,'val');
    [testX,testT] = importd(dataset,'test');
    [trainX,testX,valX] = normalize(trainX,testX,valX);    
    trainErr = zeros(length(Epsilon),1);
    valErr = zeros(length(Epsilon),1);
    testErr = zeros(length(Epsilon),1);    
    
    for i = 1:length(Epsilon)
        eps = Epsilon(i);
        [svroptions] = buildsvroptions(eps,kernel,a,b,d);
        
        [model]=c_svr_train(trainX,svroptions,trainT);
        trainET = c_svr_test(trainX,model);
        valET = c_svr_test(valX,model);
        testET = c_svr_test(testX,model);
        
    end
    
end
function [deltaEsq] = computeDeltaSquareError(Actual,Est)
     deltaEsq = deltaE .* deltaE;
     deltaEsq = sqrt(mean(deltaEsq(:)));   
end