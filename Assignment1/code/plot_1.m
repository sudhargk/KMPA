%%Plot of Actual function,Target output and Model output for Dataset 1 for different dataset size different model complexity
%% and different regularization parameters
% N  : 1 X N column vector for different dataset size
% B :  1 X N column vector for different basis values
% L : 1 X N column vector for different lambda values
% Remarks :: higher regularization needed for higher basis 
% plot_1([10 20],[4 20],exp([-25 -20]))
function [] = plot_1(N,B,L)
    [X,T] = importd('univariate','train'); 
    for i = 1:length(N)
        figure();
        idx = randperm(size(X,1),N(i));
        newX = X(idx,:);newT = T(idx);
        plotIdx =1;
        for basis = B
            xPhi = computeDesignMatrix(newX,'Polynomial',basis);
            for lambda = L
                subplot(length(B),length(L),plotIdx);
                W= train(xPhi,newT,lambda);
                %Plotting 
                Z = linspace(0,1,1000)';
                zPhi =computeDesignMatrix(Z,'Polynomial',basis);
                zY  = zPhi*W;
                zF = f(Z);
                plot(Z,zF,'g',Z,zY,'r',newX,newT,'o');
                title(['d =' num2str(basis) '  ln(\lambda) = ' num2str(log(lambda))]);
                ylim([-6 2]);
                plotIdx=plotIdx +1;
            end
        end
        
        
    end
    
end
