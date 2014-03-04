function [W] = train(PHI,T,lambda)
    numColumns = size(PHI,2);
    lambdaI = lambda*eye(numColumns);
    lambdaI(1,1) = 0;
    W = ((PHI'*PHI+lambdaI)\PHI')*T;
end