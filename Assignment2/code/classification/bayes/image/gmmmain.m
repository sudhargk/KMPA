

%%
%Implementation of bayes classifier using gaussian distribution
% @dataset = 'overlapping_data','linearlySeparableData','nonlinearlySeparable'
% @numCluster  = 1 x C  number of cluster per class
%
function []= gmmmain()
    dataset = 'image';
        path = fullfile(pwd,'..','..','..','..','data',dataset,'data');

    load(path);
    models= [18	12 6 19	10];
    numCluster = [25 18 20 20 20];
    numClass= length(models);
    [trainset,testset,classes]=split(CompleteData,models);
    [gmmObj] = train(trainset,numCluster,classes);
    [confusion]=testData(testset,gmmObj,classes);       
    [perClassInfo,overallAcc]=computeMetrics(confusion,numClass);
    format shortg;
    display(confusion);
    display(overallAcc);
    
%     visualize(testset,gmmObj);
end

function[trainset,testset,classes] = split(dataset,model)
    trainset= cell(length(model),1);
    testset=cell(length(model),1);
    classes = cell(length(model),1);
    for index =1 : length(model)
       classes{index}=dataset{model(index),2};
       idx = randperm(size(dataset{model(index),1},1));
       len = length(idx);
       trainset{index} =  dataset{model(index),1}(idx(1:0.75*len),:);
       testset{index} =  dataset{model(index),1}(idx(0.75*len+1:end),:);
    end
end

function [C]= testData(dataset,gmmObj,classes)
    numClass = size(dataset,1);
    numSample = cellfun('size',dataset,1);
    totalSample = sum(numSample);
    actualClass = getActualClass(numSample);
    dataset = cell2mat(dataset);
    [decidedClass,~]= classify(dataset,gmmObj,numClass);
    C = zeros(numClass,numClass);
    targets=full(ind2vec(actualClass));
    outputs = full(ind2vec(decidedClass));
    figure(),plotconfusion(targets,outputs),set(gcf, 'WindowStyle', 'docked');
    figure(),plotroc(targets,outputs),set(gcf, 'WindowStyle', 'docked');
    [~, ~, ph] = legend(gca);legend(ph, classes); 
    
    for i = 1:totalSample
        C(actualClass(i),decidedClass(i)) = C(actualClass(i),decidedClass(i)) + 1;
    end
end

function [decidedClass, probEst]= classify(dataset,gmmObj,numClass)
    totalSample = size(dataset,1);
    probEst = zeros(numClass, totalSample);
    for cIndex = 1 : numClass
        probEst(cIndex,:) = pdf(gmmObj{cIndex},dataset);
    end
    [~, decidedClass] = max(probEst);
end


function [C] = getActualClass(numSample)
    numClass = length(numSample);
    totalSample = sum(numSample);
    S = cumsum(numSample);
    C = ones(1,totalSample);
    A = 1:totalSample;
    for i = 1 : numClass-1
        C = C + (A > S(i));
    end
end

function [gmmObj] = train(trainsets,numCluster,classes)
    numClass = size(trainsets,1);
    gmmObj= cell(numClass,1);
    for index = 1:numClass
        display(['Building GMM model for class' classes(index) ' dataset....']);
        [gmmObj{index}] = gmdistribution.fit(trainsets{index},numCluster(index),'Regularize',0.0001,'CovType','diagonal');
    end
end