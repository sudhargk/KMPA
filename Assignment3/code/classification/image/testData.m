function [C]= testData(dataset,svm_model,actualClass,numClass)
    totalSample = size(dataset,1);
    [decidedClass]= classify(dataset,svm_model,numClass);
    targets=full(ind2vec(actualClass));
    outputs = full(ind2vec(decidedClass));
%     figure(),plotconfusion(targets,outputs),set(gcf, 'WindowStyle', 'docked');
%     figure(),plotroc(targets,outputs),set(gcf, 'WindowStyle', 'docked');
    C = zeros(numClass,numClass);
    for i = 1:totalSample
        C(actualClass(i),decidedClass(i)) = C(actualClass(i),decidedClass(i)) + 1;
    end
end