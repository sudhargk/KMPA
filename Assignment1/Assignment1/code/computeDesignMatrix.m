function[PHI]=computeDesignMatrix(X,modelType,numBasis,variance,mean)
    switch(modelType)
        case 'Polynomial'
            PHI = polynomialPhi(X,numBasis);
        case 'Gaussian'
            PHI = guassianPhi(X,numBasis,variance,mean);
        otherwise
            PHI = X;
    end
end

function [PHI] = polynomialPhi(X,numBasis)
    PHI = bsxfun(@power,X,(0:numBasis-1));
end

% M contains 1 less than numBasis means
function [PHI] = guassianPhi(X,numBasis,variance,M)
    numExamples = size(X,1);
    PHI = zeros(numExamples, numBasis);
    for kIndex = 2:numBasis;
        diff = bsxfun(@minus,X,M(kIndex-1,:));
         PHI(:,kIndex) = sum(diff.^2,2)./(2*variance);
    %PHI(:,kIndex) = (X-M(kIndex-1))*variance*(X-M(kIndex-1))'/2;
    end
    PHI = exp(-PHI); % This automatically fills first column with ones
end