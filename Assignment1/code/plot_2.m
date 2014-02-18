%% Plot of (Bias)^2, Variance, Avg Loss and Validation Data Error for Dataset 1
%  for different model complexity and different regularization parameters
%%
% D : number of different datasets to be considered
% N : number of points in each dataset
% B : 1 X P row vector for different basis values
% L : 1 X Q row vector for different lambda values
% Remarks :: higer the effective complexity, lower the bias, greater the
%            varance
% plot_2(100,25,[25],exp((-7:0.01:-2)))
function [] = plot_2(D,N,B,L)
    [trainX,trainT] = importd('univariate','train');
    [valX,valT] = importd('univariate','val');
    
    numPoints = 10000;
    step = length(trainT) / numPoints;
    idx = round((1:numPoints) * step);
    Z = trainX(idx);
    zT = trainT(idx);
            
    sqBias = zeros(length(B),length(L));
    variance = zeros(length(B),length(L));
    avgLoss = zeros(length(B),length(L));
    errVal = zeros(length(B),length(L));
    
    for i = 1:length(B)
        basis = B(i);
        zPhi = computeDesignMatrix(Z,'Polynomial',basis);
        vPhi = computeDesignMatrix(valX,'Polynomial',basis);
            
        for j = 1:length(L)
            
            lambda = L(j);
%             idx = randperm(size(trainX,1),N*D);
%             newX = reshape(trainX(idx),[N D]);
%             newT = reshape(trainT(idx), [N D]);
%             
            display(log(lambda));
            W = zeros(basis,D);
            for k = 1:D
                idx = randperm(size(trainX,1),N);
                newX = trainX(idx);
                newT = trainT(idx);
                xPhi = computeDesignMatrix(newX,'Polynomial',basis);
                W(:,k) = train(xPhi,newT,lambda);
            
%                 xPhi = computeDesignMatrix(newX(:,k),'Polynomial',basis);
%                 W(:,k) = train(xPhi,newT(:,k),lambda);
            end
            
            zY = zPhi*W;            
            avgzY = mean( zY,2 );
            
            F = f(Z);
            deltaB = avgzY - F;
            sqBias(i,j) = mean( deltaB .* deltaB );
            
            deltazY = bsxfun( @minus, zY, avgzY );
            deltazYsq = deltazY .* deltazY;
            variance(i,j) = mean(deltazYsq(:));
            
            deltaL = bsxfun( @minus, zT, zY );
            deltaLsq = deltaL .* deltaL;
            avgLoss(i,j) = mean(deltaLsq(:));
            
            vY = vPhi*W;
            deltaE = bsxfun( @minus, valT, vY );
            deltaEsq = deltaE .* deltaE;
            errVal(i,j) = mean(deltaEsq(:));
            
        end
    end
    
    % Plotting
    
    % Given lambda, plot vs number of basis functions
    if (length(B) > 1)
        lIdx = 1:length(L);
        for i = 1:length(L)
            figure,
%    if (length(B) > 1) 
%        figure(1);
%        set(1, 'WindowStyle', 'docked');
%        numPlots = 4;
%        [r, c] = configSubplots(numPlots);
%        lIdx = round(linspace(1,length(L),numPlots));
%        for i = 1:numPlots
%            subplot(r, c, i);
           plot(B', sqBias(:,lIdx(i)), 'b',...
               B', variance(:,lIdx(i)), 'r',...
               B', avgLoss(:,lIdx(i)), 'm',...
               B', errVal(:,lIdx(i)), 'k');
           title({'Squared bias, variance, average loss & error on'
               'validation data as functions of number of basis functions'
               ['ln(\lambda) = ' num2str( log( L(lIdx(i)) ) ) ]});
           ylabel('Function values');
           xlabel('Number of basis functions');
           legend('(Bias)^2', 'Variance', 'Avg loss', 'Validation error'),
           legend('boxon');
       end
    end
            
    % Given number of basis functions, plot vs lambda
    if (length(L) > 1)
        bIdx = 1:length(B);
        for i = 1:length(B)
            figure,
%     if (length(L) > 1)
%         figure(2);
%         set(2, 'WindowStyle', 'docked');
%         numPlots = 4;
%         [r, c] = configSubplots(numPlots);
%         bIdx = round(linspace(1,length(B),numPlots));
%         for i = 1:numPlots
%             subplot(r, c, i);
            plot(log(L), sqBias(bIdx(i),:), 'b',...
                log(L), variance(bIdx(i),:), 'r',...
                log(L), avgLoss(bIdx(i),:), 'm',...
                log(L), errVal(bIdx(i),:), 'k');
            title({'Squared bias, variance, average loss & error on validation'
                'data as functions of the regularization parameter \lambda'
                ['Bases = ' num2str( B(bIdx(i)) ) ]});
            ylabel('Function values');
            xlabel('ln(\lambda)');
            legend('(Bias)^2', 'Variance', 'Avg loss', 'Validation error'),
            legend('boxon');
        end
    end
end

function [r,c] = configSubplots( numPlots )
    conf = [1 1 1 2 2 2 3 3;
            1 2 3 2 3 3 3 3];
    r = conf(1,numPlots);
    c = conf(2,numPlots);
end