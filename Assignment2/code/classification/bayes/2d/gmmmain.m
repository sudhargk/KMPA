

%%
%Implementation of bayes classifier using gaussian distribution
% @dataset = 'overlapping_data','linearlySeparableData','nonlinearlySeparable'
% @numCluster  = 1 x C  number of cluster per class
%
function []= gmmmain()
    dataset = 'nonlinearlySeparable';
    path = [pwd,'\..\..\..\..\data\' dataset '\data'];
    load(path);    
    numCluster = [1 1 1];
    [gmmObj] = train(trainset,numCluster);
    [confusion]=testData(testset,gmmObj);       
    [perClassInfo,overallAcc]=computeMetrics(confusion,numClass);
    format shortg;
    display(confusion);
    display(overallAcc);
    
    visualize(testset,gmmObj);
end

function [C]= testData(dataset,gmmObj)
    numClass = size(dataset,1);
    numSample = cellfun(@length,dataset);
    totalSample = sum(numSample);
    actualClass = getActualClass(numSample);
    dataset = cell2mat(dataset);
    [decidedClass,~]= classify(dataset,gmmObj,numClass);
    C = zeros(numClass,numClass);
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

function [gmmObj] = train(trainsets,numCluster)
    numClass = size(trainsets,1);
    gmmObj= cell(numClass,1);
    for index = 1:numClass
        display(['Building GMM model for class' num2str(index) ' dataset....']);
        [gmmObj{index}] = gmdistribution.fit(trainsets{index},numCluster(index),'Regularize',0.0001,'CovType','diagonal');
    end
end

function  visualize(testset,gmmObj)
    numClass = size(testset,1);
    numSample = cellfun(@length,testset);
    AC = getActualClass(numSample);
    labels = strcat({'Class '}, num2str((1:numClass)'));
    group = ordinal(AC, labels);
    D = cell2mat(testset);
    mn = min(D); mx = max(D); n = 1000;
    
    clrLite = [1 0.6 0.6 ; 0.6 1 0.6 ; 0.6 0.6 1; 1 0.6 1];
    clrDark = [0.7 0 0 ; 0 0.7 0 ; 0 0 0.7; 0.7 0 0.7];

    [X, Y] = meshgrid( linspace(mn(1),mx(1),n), linspace(mn(2),mx(2),n) );
    Xl = X(:); Yl = Y(:);
    [gridDC, gridProb] = classify([Xl Yl], gmmObj,numClass);
    [DC, ~] = classify(D, gmmObj,numClass);
    
    %plotting all  points
    figure(1), hold on;
    image(Xl, Yl, reshape(gridDC, n, n))
    axis xy, box on, colormap(clrLite);
    
    %Superimposing data points
    gscatter(D(:,1), D(:,2), group, clrDark, '.', 15)
    
    %Superimposing wrongly classified points
    bad = (DC ~= AC);
    plot(D(bad,1), D(bad,2), 'yx', 'MarkerSize', 10)
    axis([mn(1) mx(1) mn(2) mx(2)]);
    axis equal
    xlabel('Dimension 1'), ylabel('Dimension 2'); 
    hold off   
end