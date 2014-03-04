%% Plot of Actual function,Target output and Model output for Dataset 1
%  for different training dataset sizes, different model complexity
%  and different regularization parameters
%%
% N : 1 X P row vector for different dataset size
% B : 1 X Q row vector for different basis values
% L : 1 X R row vector for different lambda values
% Remarks :: higher regularization needed for higher basis 
% plot_1([9 20],[3 9],exp([-30 -15]))
function [] = plot_1(N,B,L)
    [X,T] = importd('univariate','train');
    Z = (1:1000)' / 1000;
    zF = f(Z);
    for i = 1:length(N)
        figure(i);
        set(i, 'WindowStyle', 'docked');
        idx = randperm(size(X,1),N(i));
        newX = X(idx,:);newT = T(idx);
        plotIdx = 1;
        for basis = B
            xPhi = computeDesignMatrix(newX,'Polynomial',basis);
            zPhi = computeDesignMatrix(Z,'Polynomial',basis);
            for lambda = L
                subplot(length(B),length(L),plotIdx);
                W = train(xPhi,newT,lambda);
                
                %Plotting 
                zY = zPhi*W;
                plot(Z,zF,'g',Z,zY,'r',newX,newT,'o');
                title({['Bases = ' num2str(basis) ]
                    ['ln(\lambda) = ' num2str(log(lambda)) ]});
                ylim([-6 2]);
                plotIdx=plotIdx +1;
            end
        end
    end
end
