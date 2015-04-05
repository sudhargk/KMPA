

%%
%Implementation of bayes classifier using gaussian distribution
% @dataset = 'overlapping','linearlySeparable','nonlinearlySeparable'
% @kernel = 'linear', 'polynomial', 'gaussian'
% @cost = C-SVM Cost
% @a,@b,@d = for polynomial (a*x'.y +b)^d
% @a = for gaussian  exp(-a*|x-v|^2);
% @numCluster  = 1 x C  number of cluster per class
%
% a = 1/width = 1/0.0193; nu = 0.1481 (overlapping)

function []= nu_svdd()
    dataset = 'overlapping';
    nu = 2; a = 0.25;
    path = fullfile(pwd,'..','..','data',dataset,'data');
    load(path);
    which_class = 1;
    
%     for clIndex = 1:numClass
%         if (clIndex ~= which_class)
%             valset{clIndex} = [trainset{clIndex}; valset{clIndex}];
%         end
%     end
    
    idx = (1:numClass) ~= which_class;
    tmp = trainset;
    trainset = cell(2,1);
    trainset{1} = tmp{which_class};
    trainset{2} = cell2mat(tmp(idx));
    normal_trainset = trainset(1);
    
    tmp = valset;
    valset = cell(2,1);
    valset{1} = tmp{which_class};
    valset{2} = cell2mat(tmp(idx));
    
    tmp = testset;
    testset = cell(2,1);
    testset{1} = tmp{which_class};
    testset{2} = cell2mat(tmp(idx));
    
    numClass = 2;
    
    format shortg;
    
    nu_grid = 2.^linspace(-15, 0, 50);
    width_grid = 2.^linspace(4, -15, 50);
    acc = zeros(length(width_grid), length(nu_grid));
    for wIndex = (1:length(width_grid))
        a = width_grid(wIndex);
        acc_t = zeros(1, length(nu_grid));
        for nIndex = (1:length(nu_grid))
            nu = nu_grid(nIndex);
            svmoptions = buildSVMOptions(nu,a);
            [svm_model] = train(normal_trainset,[svmoptions ' -q']);
            [confusion] = testData(valset,svm_model, ' -q');
            acc_t(nIndex) = sum(diag(confusion)) / sum(confusion(:));
        end
        acc(wIndex, :) = acc_t;
    end
    save(['grid_search_' dataset '_nu'] , 'nu_grid', 'width_grid', 'acc');
    [~, bestacc] = max(acc(:));
    
    [i, j] = ind2sub(size(acc), bestacc);
    a = width_grid(i);
    nu = nu_grid(j);
    fprintf('=================================================================\n');
    fprintf('Parameters chosen: Gamma = %f, Nu = %f\n', a, nu);
    svmoptions = buildSVMOptions(nu, a);
    svm_model = train(normal_trainset, svmoptions);
    
    fprintf('------------------------------------------------------\n');
    fprintf('On train data:-\n');
    confusion = testData(trainset, svm_model, '');
    [perClassInfo,overallAcc] = computeMetrics(confusion,numClass);
    display(confusion);
    metrics = perClassInfo(1,:); % since we are only interested in the first class
    fprintf('Accuracy = %f (%d / %d) \n', overallAcc*100, sum(diag(confusion)), sum(sum(confusion)));
    fprintf('True positive rate = %f (%d / %d)\n', metrics(9)*100, metrics(5), metrics(1));
    fprintf('False alarm rate = %f (%d / %d)\n', metrics(10)*100, metrics(7), metrics(2));
    
    fprintf('------------------------------------------------------\n');
    fprintf('On validation data:-\n');
    confusion = testData(valset, svm_model, '');
    [perClassInfo,overallAcc] = computeMetrics(confusion,numClass);
    display(confusion);
    metrics = perClassInfo(1,:); % since we are only interested in the first class
    fprintf('Accuracy = %f (%d / %d) \n', overallAcc*100, sum(diag(confusion)), sum(sum(confusion)));
    fprintf('True positive rate = %f (%d / %d)\n', metrics(9)*100, metrics(5), metrics(1));
    fprintf('False alarm rate = %f (%d / %d)\n', metrics(10)*100, metrics(7), metrics(2));
    
    fprintf('------------------------------------------------------\n');
    fprintf('On test data:-\n');
    confusion = testData(testset, svm_model, '', true, ['Overlapping Test Data - Nu SVM - nu = ' num2str(nu) ', width = ' num2str(a)]);
    [perClassInfo,overallAcc] = computeMetrics(confusion,numClass);
    display(confusion);
    metrics = perClassInfo(1,:); % since we are only interested in the first class
    fprintf('Accuracy = %f (%d / %d) \n', overallAcc*100, sum(diag(confusion)), sum(sum(confusion)));
    fprintf('True positive rate = %f (%d / %d)\n', metrics(9)*100, metrics(5), metrics(1));
    fprintf('False alarm rate = %f (%d / %d)\n', metrics(10)*100, metrics(7), metrics(2));
    
    visualize(normal_trainset,svm_model,1);
end

function [svmoptions] = buildSVMOptions(nu,gamma)
    soptions ='-s 2';
    koptions = [' -t 2 -g ' num2str(gamma)];
    noptions = ['-n ' num2str(nu)];
    svmoptions = [soptions ' ' koptions ' ' noptions];
end

function [svm_model] = train(trainsets,svmoptions)
    numSample = cellfun(@length,trainsets);
    classLabels = getActualClass(numSample)';
    trainsets = cell2mat(trainsets);
    svm_model = svmtrain(classLabels,trainsets,svmoptions);
end