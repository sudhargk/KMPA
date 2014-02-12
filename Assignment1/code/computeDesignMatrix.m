function [PHI] = computeDesignMatrix(X,modelType,numBasis,variance)
    switch(modelType)
        case 'Polynomial'
            PHI = polynomialDesignMatrix(X,numBasis);
        case 'Guassian'
            PHI = guassianDesignMatrix(X,numBasis,variance);
        otherwise
            PHI = X;
    end   
end

function [PHI] = polynomialDesignMatrix(X,numBasis)
    PHI = bsxfun(@power,X,(0:numBasis-1));
end
function [PHI] = guassianDesignMatrix(X,numBasis,variance)
    numExamples = size(X,1);
    [~,M]  = kmeans(X,numBasis);
    PHI = zeros(numExamples, numBasis);
    for kIndex = 1:numBasis;
        PHI(:,kIndex) = sum((X-M(kIndex)).^2,2)./(2*variance);
    end
    PHI = exp(-PHI);
end