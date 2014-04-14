function [decidedClass]= classify(dataset,svm_model,actualClass, svm_options)
    actualClass( actualClass~=1 ) = -1;
    [decidedClass] = svmpredict(actualClass,dataset,svm_model,svm_options); 
    decidedClass = decidedClass';
    decidedClass( decidedClass==-1 ) = 2;
end