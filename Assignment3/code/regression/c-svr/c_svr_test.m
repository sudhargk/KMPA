function [estimated_output] = c_svr_test(dataset,svm_model)
    numSamples = size(dataset,1);
    vagueoutputs = randn(numSamples,1);
    estimated_output = svmpredict(vagueoutputs,dataset,svm_model);
end