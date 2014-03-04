%% PERCEPTRON Perceptron learning implementation
%   Implements a multi-category perceptron classifier
%   @params - whichDataset : choice of dataset 'overlapping_data','linearlySeparableData','nonlinearlySeparable'
%   @params - choice : the learning method used. Can be
%             'regular' - vanilla perceptron (only for comparison purposes)
%             'mira' - MIRA (Margin infused relaxed algorithm)
%            - init - whether to initialize the weights randomly or with all zeros
%           - avgWts - boolean indicating whether final weight vector is chosen as is, OR
%                      averaging of weight vectors is done to prevent thrashing
%           
%
%%

function [] = perceptron(whichDataset, choice, init, avgWts, maxIterations, minDiff )
%     whichDataset = 'nonlinearlySeparable';
    whichDataset = [pwd,'\..\..\..\data\' whichDataset '\data'];
  
    if (~exist('minDiff', 'var'))
        minDiff = eps;
    end
    if (~exist('maxIterations', 'var'))
        maxIterations = 500;
    end
    if (~exist('choice', 'var'))
        choice = 'mira';
    end
    if (~exist('init', 'var'))
        init = 'zeros';
    end
    if (~exist('avgWts', 'var'))
        avgWts = true;
    end
    load(whichDataset);
%     trainset = mergeClassData(trainset);
    weights = trainData(trainset, choice, init, avgWts, maxIterations, minDiff);
    confusion = testData(testset, weights);
    [~,overallAcc]=computeMetrics(confusion,numClass);
    display(confusion);
    display(overallAcc);
end

function [W] = trainData(dataset, choice, init, avgWts, maxIterations, minDiff)
    numClass = length(dataset);
    numSample = cellfun(@length, dataset);
    D = cell2mat(dataset);
    
    % Initialization
    numDim = size(D, 2);
    D = [ones(length(D),1) D]; % D - augmented feature vectors
    if (strcmp(init, 'random'))
        W = rand(numClass, numDim+1);    % W - augmented weight vectors
    elseif (strcmp(init, 'zeros'))
        W = zeros(numClass, numDim+1);    % W - augmented weight vectors
    else
        error('Invalid choice for initialization');
    end
    
    AC = getActualClass(numSample);
    
    i = 1;
    diff = inf;
    AVGW = zeros(numClass, numDim+1);
    while (diff > minDiff && i <= maxIterations)
%        fprintf('Iteration %d\n', i);
%         display(W');
        oldW = W;
        scores = W * D';
        [~, DC] = max(scores);
        bad = find(AC ~= DC);
        if (strcmp(choice,'regular'))
            for j = 1 : length(bad)
                ind = bad(j);
                W(AC(ind),:) = W(AC(ind),:) + D(ind,:);
                W(DC(ind),:) = W(DC(ind),:) - D(ind,:);
            end
        elseif (strcmp(choice,'mira'))
            for j = 1 : length(bad)
                ind = bad(j);
                scorediff = scores(DC(ind), ind) - scores(DC(ind), ind);
                k = (scorediff + 1) / (2 * (D(ind,:) * D(ind,:)'));
                tau = min(k, 5);
                W(AC(ind),:) = W(AC(ind),:) + tau * D(ind,:);
                W(DC(ind),:) = W(DC(ind),:) - tau * D(ind,:);
            end
        else
            error('Invalid algorithmic choice');
        end
        
        if (avgWts == true)
            oldAVGW = AVGW;
            AVGW = (i/(i+1)) * AVGW + (1/(i+1)) * W;
            diff = norm(AVGW - oldAVGW);
        else
            diff = norm(W - oldW);
        end
        i = i + 1;
    end
    display(diff);
    if (avgWts == true)
        W = AVGW;
    end
end


function [C]= testData(dataset, weights)
    numClass = length(dataset);
    C = zeros(numClass);
    numClass = size(dataset,1);
    numSample = cellfun(@length,dataset);
    totalSample = sum(numSample);
    actualClass = getActualClass(numSample);
    dataset = cell2mat(dataset);
    [decidedClass]= classify(dataset,weights,numClass);
    C = zeros(numClass,numClass);
    for i = 1:totalSample
        C(actualClass(i),decidedClass(i)) = C(actualClass(i),decidedClass(i)) + 1;
    end
end

function[decidedClass]= classify(dataset,weights,numClass)
    totalSample = size(dataset,1);
%     probEst = zeros(numClass, totalSample);
    dataset = [ones(totalSample,1) dataset];
    scores = weights * dataset';
    [~, decidedClass]= max(scores);
%        freq = histc(decidedClasses, 1:numClass);
%         probEst(cIndex,:) = freq ./ totalSample;      
   
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