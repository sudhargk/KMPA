%% Plot of RMS error on train, test and validation data
%  for different model complexity and different regularization parameters
%   plot_2('univariate',1,9,0,[0.1:0.1:1])
function [] = plot_2(dataset,cost,gamma,coef,eps,NU)

    
%    kernel =  'gaussian';
     if(nargin<6)
        dataset = 'bivariate';
        cost = 1; gamma = 9; coef=0; eps=0.4; NU=[0.8:0.01:1];
     end
    [trainX,trainT] = importd(dataset,'train');
    [valX,valT] = importd(dataset,'val');
    [testX,testT] = importd(dataset,'test');
    [trainX,testX,valX] = normalize(trainX,testX,valX);    
    trainErr = zeros(length(NU),1);
    valErr = zeros(length(NU),1);
    testErr = zeros(length(NU),1);    
    
    for i = 1:length(NU)
        nu = NU(i);
        [svroptions] = buildsvroptions(cost,coef,gamma,eps,nu);
        
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
    generatePlot(NU, trainErr, 'Error on trainning Data');
    generatePlot(NU, valErr, 'Error on validation Data');
    generatePlot(NU, testErr, 'Error on test Data');
end
function [deltaEsq] = computeDeltaSquareError(Actual,Est)
     deltaEsq = deltaE .* deltaE;
     deltaEsq = sqrt(mean(deltaEsq(:)));   
end

function [] = generatePlot(NU, Error, type)
    figure();
    hold on;
    plot(NU,Error,'-');
    title('Plot of \nu Vs MSE');
    xlabel('\nu');
    ylabel(type);
    hold off;
    
end
