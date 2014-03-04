%% Comparison Plot of Bias and Variance for Dataset 1
%  for different model complexity and different regularization parameters
%%
% D : number of different datasets to be considered
% N : number of points in each dataset
% B : 1 X P row vector for different basis values
% L : 1 X Q row vector for different lambda values
% Remarks :: higer the effective complexity, lower the bias, greater the
%            varance
% plot_1_2(100,25,[4 25],exp([-25 -5]))
function [] = plot_1_2(D,N,B,L)
    [trainX,trainT] = importd('univariate','train');
    Z = (1:1000)' / 1000;
    
    modelNo = 1;    
    for basis = B
        zPhi = computeDesignMatrix(Z,'Polynomial',basis);
        for lambda = L
            idx = randperm(size(trainX,1),N*D);
            newX = reshape(trainX(idx),[N D]);
            newT = reshape(trainT(idx), [N D]);
            
            figure(modelNo);
            set(modelNo, 'WindowStyle', 'docked');
            
            W = zeros(basis,D);
            for k = 1:D
                xPhi = computeDesignMatrix(newX(:,k),'Polynomial',basis);
                W(:,k) = train(xPhi,newT(:,k),lambda);
            end
            
            %Plotting
            zY = zPhi*W;
            
            subplot(1,2,1);
            nplots = 20;
            idx = randperm(D, nplots);
            plot(Z,zY(:,idx),'r');
            text(0.4, -4, {['Bases = ' num2str(basis)] ...
                ['ln(\lambda) = ' num2str(log(lambda))]});
            title({'Model output functions estimated using '
                [num2str(nplots) ' different training datasets']});
            ylabel('y(x,$\overline{w}^*$)', 'Interpreter', 'latex',...
                'FontSize', 14);
            xlabel('x');
            ylim([-6 2]);
            
            subplot(1,2,2);
            avgzY = mean(zY,2);
            plot(Z, f(Z), 'g', Z, avgzY, 'r');
            text(0.4, -4, {['Bases = ' num2str(basis)]
                ['ln(\lambda) = ' num2str(log(lambda))]});
            title({['Average model output over ' num2str(D) ' datasets (red)']
                'along with actual generating function (green)'});
            ylabel('Function value');
            xlabel('x');
            ylim([-6 2]);
            
            modelNo = modelNo + 1;
        end
    end
end