
function [svm_model] = train(trainsets,svmoptions,classLabels)
    svm_model = ovrtrain(classLabels',trainsets,svmoptions);
end
