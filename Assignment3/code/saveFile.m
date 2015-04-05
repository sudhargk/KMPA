function [] = saveFile()
%     saveAll('linearlySeparableData');
%     saveAll('nonlinearlySeparable');
%     saveAll('overlapping_data');
    saveImageData();
end

function []=saveAll(dataset)
    trainset = loadData(dataset,'train');
    testset = loadData(dataset,'test');
    valset = loadData(dataset,'val');
    numClass = size(trainset,1);
    save(fullfile(pwd, '..', 'data', dataset, 'data'), 'trainset', 'testset','valset','numClass');
end

function []=saveImageData()
    givenClasses = [18 12 6 19 10];
    dataset = 'image';
    path = fullfile(pwd, '..', 'data', dataset, 'CompleteData.mat');
    load(path);
    
    data = CompleteData(givenClasses,:);
    classes = data(:,2);
    numClass = size(data, 1);
    trainset = cell(numClass,1);
    valset = cell(numClass,1);
    testset = cell(numClass,1);
    trainratio = 0.7;
    valratio = 0.1;
    
    for cIndex = 1:numClass
        
        examples = data{cIndex,1};
        numExamples = size(examples, 1);
        idx = randperm(numExamples);
        
        numTrain = floor(trainratio*numExamples);
        numVal = floor(valratio*numExamples);
        numTest = numExamples - numTrain - numVal;
   
        b = 1; e = numTrain;
        trainset{cIndex} = examples(idx(b:e), :);
        totalPixels = sum(trainset{cIndex},2);
        trainset{cIndex}=bsxfun(@rdivide,trainset{cIndex},totalPixels);
        b = e + 1; e = e + numVal;
        valset{cIndex,1} = examples(idx(b:e), :);
        totalPixels = sum(valset{cIndex},2);
        valset{cIndex}=bsxfun(@rdivide,valset{cIndex},totalPixels);
        b = e + 1; e = e + numTest;
        testset{cIndex} = examples(idx(b:e), :);
        totalPixels = sum(testset{cIndex},2);
        testset{cIndex}=bsxfun(@rdivide,testset{cIndex},totalPixels);
        
    end
    save(fullfile(pwd, '..', 'data', dataset, 'data'), 'trainset', 'testset','valset','numClass', 'classes');
end

function [dataset] = loadData(dataset,type)
    
    path = fullfile(pwd,'..', 'data', dataset);
    files = dir(fullfile(path, 'class*_', type, '.txt'));
    dataset = cell(size(files,1),1);
    index =1;
    for file = files'
        dataset{index}=importdata(fullfile(path, file.name));
        index=index+1;
    end
<<<<<<< HEAD
end
=======
end
>>>>>>> 1a389ff681b826be7e7528f6b64a5811083343d4
