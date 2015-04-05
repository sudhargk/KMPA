function [decidedClass]= classify(dataset,svm_model,numClass)
    totalSample = size(dataset,1);
    testingLabels  =  floor((1:totalSample)/ceil(totalSample/numClass))+1;
    testingLabels = testingLabels';
    [decidedClass] = ovrpredict(testingLabels,dataset,svm_model); 
    decidedClass = decidedClass';
<<<<<<< HEAD
end
=======
end
>>>>>>> 1a389ff681b826be7e7528f6b64a5811083343d4
