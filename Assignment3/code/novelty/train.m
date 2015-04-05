function [svm_model] = train(trainsets,svmoptions)
    numSample = cellfun(@length,trainsets);
    classLabels = getActualClass(numSample)';
    trainsets = cell2mat(trainsets);    
    svm_model = ovrtrain(classLabels,trainsets,svmoptions);
end