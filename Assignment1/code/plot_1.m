
function [] =plot_1()
    [X,T] = import('univariate','train'); 
    n = [50];
    for i = 1:length(n)
        idx = randperm(size(X,1),n(i));
        newX = X(idx,:);newT = T(idx);
        for basis = [50]
            xPhi = computeDesignMatrix(newX,'Polynomial',basis);
            for lambda = exp([-Inf -20 0])
                W= train(xPhi,newT,lambda);
                %Plotting 
                Z = linspace(0,1,1000)';
                zPhi =computeDesignMatrix(Z,'Polynomial',basis);
                zY  = zPhi*W;
                zF = f(Z);
                plot(Z,zF,'g',Z,zY,'r',newX,newT,'o');
                ylim([-6 2]);
            end
        end
        
        
    end
    
end
function [X,T] = import(type, filetype)
    t = [pwd '\..\data\' type '\' filetype '.txt'];
    Z = importdata(t);
    X = Z(:,1:size(Z,2)-1);
    T = Z(:,size(Z,2));
end