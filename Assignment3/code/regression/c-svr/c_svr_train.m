function [svm_model] = c_svr_train(trainsets,svmoptions,outputs)
    svm_model = svmtrain(outputs,trainsets,svmoptions);
end