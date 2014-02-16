%% Plot of RMS error on train, test and validation data
%  for different model complexity and different regularization parameters
%%
% N : number of points in training set to be taken
% B : 1 X P row vector for different basis values
% L : 1 X Q row vector for different lambda values
% dataset : type of dataset - univariate etc.
% Remarks :: Plotting RMS error rather than MSE
% Eg.
%       plot_3(9,[4],exp([-30:0.1:-6]),'univariate','Polynomial',0.3)
function [] = plot_3(N,B,L,dataset,modeltype,variance)
    
    [trainX,trainT] = importd(dataset,'train');
    [valX,valT] = importd(dataset,'val');
    [testX,testT] = importd(dataset,'test');
    [trainX,testX,valX] = normalize(trainX,testX,valX);    
    trainErr = zeros(length(B),length(L));
    valErr = zeros(length(B),length(L));
    testErr = zeros(length(B),length(L));
    
    idx = randperm( size(trainX,1), N);
    trainX = trainX(idx,:);
    trainT = trainT(idx);
    
    for i = 1:length(B)
        basis = B(i);
        
%         if (strcmp(modeltype,'Gaussian') == 0)
            [M,~] = computeClusterMeans(trainX,basis);
%         end
        trainPhi = computeDesignMatrix(trainX,modeltype,basis,variance,M);
        testPhi = computeDesignMatrix(testX,modeltype,basis,variance,M);
        valPhi = computeDesignMatrix(valX,modeltype,basis,variance,M);
            
        for j = 1:length(L)
            
            lambda = L(j);
            W = train(trainPhi,trainT,lambda);
            
            trainY = trainPhi*W;
            deltaE = (trainT - trainY);
            deltaEsq = deltaE .* deltaE;
            trainErr(i,j) = sqrt(mean(deltaEsq(:)));
            
            valY = valPhi*W;
            deltaE = (valT - valY);
            deltaEsq = deltaE .* deltaE;
            valErr(i,j) = sqrt(mean(deltaEsq(:)));
            
            testY = testPhi*W;
            deltaE = (testT - testY);
            deltaEsq = deltaE .* deltaE;
            testErr(i,j) = sqrt(mean(deltaEsq(:)));
            
        end
    end
    
    % Plotting
    
    % Given lambda, plot vs number of basis functions
%     figure(1);
%     set(1, 'WindowStyle', 'docked');
%     numPlots = 4;
%     [r, c] = configSubplots(numPlots);
%     lIdx = round(linspace(1,length(L),numPlots));
%     for i = 1:numPlots
%         subplot(r, c, i);
    if (length(B) > 1)
        lIdx = 1:length(L);
        for i = 1:length(L)
            figure,
            set(gcf, 'WindowStyle', 'docked');
            plot(B', trainErr(:,lIdx(i)), 'b',...
                B', valErr(:,lIdx(i)), 'r',...
                B', testErr(:,lIdx(i)), 'm');
            %         text(0.4, -4, ['ln(\lambda) = ' num2str( log( L(lIdx(i)) ) ) ]);
            title({'RMS error on train, test and validation data'
                'as functions of the number of basis functions'
                ['ln(\lambda) = ' num2str( log( L(lIdx(i)) ) ) ]});
            ylabel('E_{RMS}');
            xlabel('Number of basis functions');
%             err = [trainErr(:,lIdx(i)); valErr(:,lIdx(i)); testErr(:,lIdx(i))];
%             ymin = min(err);
%             ymax = max(err);
%             ylim([0 min([20*ymin ymax])]);
            legend('Train error', 'Validation error', 'Test error');
        end
    end
            
    % Given number of basis functions, plot vs lambda
%     figure(2);
%     set(2, 'WindowStyle', 'docked');
%     numPlots = 4;
%     [r, c] = configSubplots(numPlots);
%     bIdx = round(linspace(1,length(B),numPlots));
%     for i = 1:numPlots
%         subplot(r, c, i);
    if (length(L) > 1)
        bIdx = 1:length(B);
        for i = 1:length(B)
            figure,
            set(gcf, 'WindowStyle', 'docked');
            plot(log(L), trainErr(bIdx(i),:), 'b',...
                log(L), valErr(bIdx(i),:), 'r',...
                log(L), testErr(bIdx(i),:), 'm');
            %         text(0.4, -4, ['Bases = ' num2str( B(bIdx(i)) ) ]);
            title({'RMS error on train, test and validation data'
                'as functions of the regularization parameter \lambda'
                ['Bases = ' num2str( B(bIdx(i)) ) ]});
            ylabel('E_{RMS}');
            xlabel('ln(\lambda)');
%             err = [trainErr(bIdx(i),:) valErr(bIdx(i),:) testErr(bIdx(i),:)];
%             ymin = min(err);
%             ymax = max(err);
%             ylim([0 min([20*ymin ymax])]);
            legend('Train error', 'Validation error', 'Test error');
            legend('Train error', 'Validation error', 'Test error');
        end
    end
end

function [r,c] = configSubplots( numPlots )
    conf = [1 1 1 2 2 2 3 3;
            1 2 3 2 3 3 3 3];
    r = conf(1,numPlots);
    c = conf(2,numPlots);
end