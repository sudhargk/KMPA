function[PHI]=computePhiFromDesign(X,modelType,numBasis,variance,mean)
    switch(modelType)
        case 'Polynomial'
            PHI = polynomialPhi(X,numBasis);
        case 'Guassian'
            PHI = guassianPhi(X,numBasis,variance,mean);
        otherwise
            PHI = X;
    end
end

function [PHI] = polynomialPhi(X,numBasis)
    PHI = bsxfun(@power,X,(0:numBasis-1));
end

function [PHI] = guassianPhi(X,numBasis,variance,M)
    numExamples = size(X,1);
    PHI = zeros(numExamples, numBasis);
    for kIndex = 1:numBasis;
        PHI(:,kIndex) = sum((X-M(kIndex)).^2,2)./(2*variance);
    end
    PHI = exp(-PHI);
end