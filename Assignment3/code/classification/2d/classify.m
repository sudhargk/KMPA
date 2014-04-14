function [decidedClass]= classify(dataset,svm_model,numClass)
    totalSample = size(dataset,1);
    testingLabels  =  floor((1:totalSample)/ceil(totalSample/numClass))+1;
    testingLabels = testingLabels';
    [decidedClass] = ovrpredict(testingLabels,dataset,svm_model); 
    decidedClass = decidedClass';
end
