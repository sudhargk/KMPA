

%%
%Implementation of bayes classifier using gaussian distribution
% @dataset = 'overlapping','linearlySeparable','nonlinearlySeparable'
% @kernel = 'linear', 'polynomial', 'gaussian'
% @cost = C-SVM Cost
% @a,@b,@d = for polynomial (a*x'.y +b)^d
% @a = for gaussian  exp(-a*|x-v|^2);
% @numCluster  = 1 x C  number of cluster per class
%


function []= c_svdd_multivariate()
    dataset = 'novelty';
    cost = 2; a = 0.25;
    path = fullfile(pwd,'..','..','data',dataset,'data');
    load(path);
    which_class = 1;
    for clIndex = 1:numClass
%         if (clIndex ~= which_class)
%             valset{clIndex} = [trainset{clIndex}; valset{clIndex}];
            valset{clIndex} = valset{clIndex};
%         end
    end
    
    normal_trainset = trainset(which_class);
    
    format shortg;
    
    cost_grid = 2.^(-2:2);
    width_grid = 2.^linspace(0, -8, 5);
    acc = zeros(length(width_grid), length(cost_grid));
    mcc = zeros(length(width_grid), length(cost_grid));
    parfor wIndex = (1:length(width_grid))
        a = width_grid(wIndex);
        acc_t = zeros(1, length(cost_grid));
        mcc_t = zeros(1, length(cost_grid));
        for cIndex = (1:length(cost_grid))
            cost = cost_grid(cIndex);
            svmoptions = buildSVMOptions(cost,a);
            [svm_model] = train(normal_trainset,[svmoptions ' -q']);
            [confusion] = testData(valset,svm_model,'-q');
            acc_t(cIndex) = sum(diag(confusion)) ./ sum(confusion(:));
            tp = confusion(1,1);
            tn = confusion(2,2);
            fn = confusion(1,2);
            fp = confusion(2,1);
            numerator = tp*tn - fp*fn;
            denominator = sqrt((tp+fp)*(fn+tn)*(tp+fn)*(fp+tn));
            if denominator == 0
                denominator = 1;
            end
            mcc_t(cIndex) = numerator / denominator;
        end
        acc(wIndex, :) = acc_t;
        mcc(wIndex, :) = mcc_t;
    end
    
    save(['grid_search_' dataset '_c'] , 'cost_grid', 'width_grid', 'acc', 'mcc');
    [~, bestacc] = max(acc(:));
    [~, bestmcc] = max(mcc(:));
    if (bestmcc ~= bestacc)
        fprintf('Parameter choice with best MCC does not give best ACC! Classes may be imbalanced');
    end
    [i, j] = ind2sub(size(mcc), bestmcc);
    a = width_grid(i);
    cost = cost_grid(j);
    fprintf('=================================================================\n');
    fprintf('Parameters chosen: Gamma = %f, C = %f\n', a, cost);
    svmoptions = buildSVMOptions(cost, a);
    svm_model = train(normal_trainset, svmoptions);
    
    fprintf('------------------------------------------------------\n');
    fprintf('On train data:-\n');
    confusion = testData(trainset, svm_model, '', true);
    [perClassInfo,overallAcc] = computeMetrics(confusion,numClass);
    display(confusion);
    metrics = perClassInfo(1,:); % since we are only interested in the first class
    fprintf('Accuracy = %f (%d / %d) \n', overallAcc*100, sum(diag(confusion)), sum(sum(confusion)));
    fprintf('True positive rate = %f (%d / %d)\n', metrics(9)*100, metrics(5), metrics(1));
    fprintf('False alarm rate = %f (%d / %d)\n', metrics(10)*100, metrics(7), metrics(2));
    
    fprintf('------------------------------------------------------\n');
    fprintf('On validation data:-\n');
    confusion = testData(valset, svm_model, '', true);
    [perClassInfo,overallAcc] = computeMetrics(confusion,numClass);
    display(confusion);
    metrics = perClassInfo(1,:); % since we are only interested in the first class
    fprintf('Accuracy = %f (%d / %d) \n', overallAcc*100, sum(diag(confusion)), sum(sum(confusion)));
    fprintf('True positive rate = %f (%d / %d)\n', metrics(9)*100, metrics(5), metrics(1));
    fprintf('False alarm rate = %f (%d / %d)\n', metrics(10)*100, metrics(7), metrics(2));
    
    fprintf('------------------------------------------------------\n');
    fprintf('On test data:-\n');
    confusion = testData(testset, svm_model, '', true, ['Multivariate Test Data - C SVM - C = ' num2str(cost) ', width = ' num2str(a)]);
    [perClassInfo,overallAcc] = computeMetrics(confusion,numClass);
    display(confusion);
    metrics = perClassInfo(1,:); % since we are only interested in the first class
    fprintf('Accuracy = %f (%d / %d) \n', overallAcc*100, sum(diag(confusion)), sum(sum(confusion)));
    fprintf('True positive rate = %f (%d / %d)\n', metrics(9)*100, metrics(5), metrics(1));
    fprintf('False alarm rate = %f (%d / %d)\n', metrics(10)*100, metrics(7), metrics(2));
    
end

function [svmoptions] = buildSVMOptions(cost,gamma)
    soptions ='-s 5';
    koptions = [' -t 2 -g ' num2str(gamma)];
    coptions = ['-c ' num2str(cost)];
    svmoptions = [soptions ' ' koptions ' ' coptions];
end

function [svm_model] = train(trainsets,svmoptions)
    numSample = cellfun(@length,trainsets);
    classLabels = getActualClass(numSample)';
    trainsets = cell2mat(trainsets);
    svm_model = svmtrain(classLabels,trainsets,svmoptions);
end