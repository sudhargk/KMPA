function [C]= testData(dataset,svm_model,svmoptions, display, datatype)
    if (nargin < 5)
        display = false;
    end
    numClass = size(dataset,1);
    numSample = cellfun(@length,dataset);
    totalSample = sum(numSample);
    actualClass = getActualClass(numSample)';
    dataset = cell2mat(dataset);
    [decidedClass] = classify(dataset,svm_model,actualClass, svmoptions);
    if (display == true)
        targets=full(ind2vec(actualClass'));
        outputs = full(ind2vec(decidedClass));
        figure, set(gcf, 'WindowStyle', 'docked'), plotconfusion(targets,outputs), title(datatype);
        figure, set(gcf, 'WindowStyle', 'docked'), plotroc(targets,outputs), title(datatype);
    end
    C = zeros(numClass,numClass);
    for i = 1:totalSample
        C(actualClass(i),decidedClass(i)) = C(actualClass(i),decidedClass(i)) + 1;
    end
end