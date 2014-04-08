function [X,T,classes] = importd(type, filetype)
    persistent F;
    persistent D;
    if (strcmp(F,type) == 0)
        F = type;
        t = fullfile(pwd, '..', '..', '..' , 'data', type, 'data.mat');
        D = load(t);
    end
    switch (filetype)
        case 'train'
            classExamples = cellfun(@(x)(size(x,1)), D.trainset);
            numClasses = size(D.trainset,1);
            X = cell2mat(D.trainset)';
        case 'test'
            classExamples = cellfun(@(x)(size(x,1)), D.testset);
            numClasses = size(D.testset,1);
            X = cell2mat(D.testset)';
        case 'val'
            classExamples = cellfun(@(x)(size(x,1)), D.valset);
            numClasses = size(D.valset,1);
            X = cell2mat(D.valset)';
    end
    numExamples = sum(classExamples);
    T = zeros(numClasses, numExamples);
    % 1-of-K representation
    b = 1;
    for i = 1:numClasses
        e = b + classExamples(i);
        T(i,b:e-1) = 1;
        b = e;
    end
    if (strcmp(type, 'image')==1)
        classes = D.classes;
    end
end