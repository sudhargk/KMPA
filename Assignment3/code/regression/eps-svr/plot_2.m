%% Plot of RMS error on train, test and validation data
%  for different model complexity and different regularization parameters
%   plot_2('univariate',1,9,0,[0.1:0.1:1])
function [] = plot_2(dataset,cost,gamma,coef,Epsilon)

    
%    kernel =  'gaussian';
     if(nargin<5)
        dataset = 'bivariate';
        cost = 1; gamma = 9; coef=0; Epsilon=[0.1:0.1:1]; 
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
        [svroptions] = buildsvroptions(cost,coef,gamma,eps);
        
        [model]=c_svr_train(trainX,svroptions,trainT);
        trainET = c_svr_test(trainX,model);
        valET = c_svr_test(valX,model);
        testET = c_svr_test(testX,model);
        trainErr(i,1) = ((trainET - trainT)' * (trainET - trainT)) ./ length(trainT);
        valErr(i,1) = ((valET - valT)' * (valET - valT)) ./ length(testT);
        testErr(i,1) = ((testET - testT)' * (testET - testT)) ./ length(testT);
        
    end
    trainErr
    valErr
    testErr
    generatePlot(Epsilon, trainErr, 'Error on trainning Data');
    generatePlot(Epsilon, valErr, 'Error on validation Data');
    generatePlot(Epsilon, testErr, 'Error on test Data');
    
end
function [deltaEsq] = computeDeltaSquareError(Actual,Est)
     deltaEsq = deltaE .* deltaE;
     deltaEsq = sqrt(mean(deltaEsq(:)));   
end

function [] = generatePlot(Epsilon, Error, type)
    figure;
    set(gcf, 'WindowStyle', 'docked');
    hold on;
    plot(Epsilon,Error,'-');
    title('Plot of \epsilon Vs MSE');
    xlabel('\epsilon');
    ylabel(type);
    hold off;
    
end
