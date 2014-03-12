function [W] = train(PHI,T,lambda,tichonovDist)
    numColumns = size(PHI,2);
    lambdaI = eye(numColumns);
    lambdaI(2:end,2:end) = lambda*tichonovDist;
    W = ((PHI'*PHI+lambdaI)\PHI')*T;
end