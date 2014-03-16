function [classLabels] = getActualClass(numSample)
    %GETACTUALCLASS Actual class label computation
    %   Computes the actual class label vector for the entire data given
    %   the number of examples in each class. Assumes that the dataset
    %   contains all the examples of one class followed by the next and so
    %   on.
    %   @params - numSample : Vector containing number of examples in each
    %   class

    numClass = length(numSample);
    totalSample = sum(numSample);
    S = cumsum(numSample);
    classLabels = ones(1,totalSample);
    A = 1:totalSample;
    for i = 1 : numClass-1
        classLabels = classLabels + (A > S(i));
    end
end


