
function [C] = getActualClass(numSample)
    numClass = length(numSample);
    totalSample = sum(numSample);
    S = cumsum(numSample);
    C = ones(1,totalSample);
    A = 1:totalSample;
    for i = 1 : numClass-1
        C = C + (A > S(i));
    end
<<<<<<< HEAD
end
=======
end
>>>>>>> 1a389ff681b826be7e7528f6b64a5811083343d4
