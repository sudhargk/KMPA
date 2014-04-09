

%%
%Implementation of bayes classifier using gaussian distribution
% @dataset = 'overlapping','linearlySeparable','nonlinearlySeparable'
% @kernel = 'linear', 'polynomial', 'gaussian'
% @cost = C-SVM Cost
% @a,@b,@d = for polynomial (a*x'.y +b)^d  
% @a = for gaussian  exp(-a*|x-v|^2);
% @numCluster  = 1 x C  number of cluster per class
%
function []= c_svm(dataset,kernel,cost,a,b,d)
     if(nargin<6)
        dataset = 'linearlySeparable';
        kernel =  'polynomial';
        cost = 1; a = 4; b = 3; d = 2; 
     end
    path = fullfile(pwd,'..','..','data',dataset,'data');
    
    load(path);    
   
    svmoptions = buildSVMOptions(cost,kernel,a,b,d);
    buildKernelGram(cell2mat(trainset),cell2mat(trainset),kernel,a,b,d);
    [svm_model] = train(trainset,svmoptions);
    [confusion]=testData(testset,svm_model);       
    [perClassInfo,overallAcc]=computeMetrics(confusion,numClass);
     format shortg;
     display(confusion);
     display(overallAcc);
      visualize(trainset,svm_model,cost);
      
end
function [svmoptions] = buildSVMOptions(cost,kernel,gamma,coef,degree)
    soptions ='-s 0';
    koptions = '-t';
    switch(kernel)
        case 'linear'  
            koptions = [koptions ' 0'];
        case 'polynomial'
            koptions = [koptions ' 1'];
            koptions = [koptions ' -g ' num2str(gamma)];
            koptions = [koptions ' -r ' num2str(coef)];
            koptions = [koptions ' -d ' num2str(degree)];
        case 'gaussian'
            koptions = [koptions ' 2'];
            koptions = [koptions ' -g ' num2str(gamma)];
    end
    coptions = ['-c ' num2str(cost)];
    boptions = '-b 1';
    svmoptions = [soptions ' ' koptions ' ' coptions,' ',boptions];
end
function [C]= testData(dataset,svm_model)
    numClass = size(dataset,1);
    numSample = cellfun(@length,dataset);
    totalSample = sum(numSample);
    actualClass = getActualClass(numSample);
    dataset = cell2mat(dataset);
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

function [decidedClass]= classify(dataset,svm_model,numClass)
    totalSample = size(dataset,1);
    testingLabels  =  floor((1:totalSample)/ceil(totalSample/numClass))+1;
    testingLabels = testingLabels';
    [decidedClass] = ovrpredict(testingLabels,dataset,svm_model); 
    decidedClass = decidedClass';
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

function [svm_model] = train(trainsets,svmoptions)
    numSample = cellfun(@length,trainsets);
    classLabels = getActualClass(numSample)';
    trainsets = cell2mat(trainsets);    
    svm_model = ovrtrain(classLabels,trainsets,svmoptions);
end

function [bvectors,ubvectors] =  computeSupportVectors(svm_model,cost)
    numModel = length(svm_model.models);
     bvectors = [];
     ubvectors = [];
    for mIndex = 1:numModel
        logicals =  ((svm_model.models{mIndex}.sv_coef-cost)<eps);
        bvectors = [bvectors; svm_model.models{mIndex}.sv_indices(logicals)];
        ubvectors = [ubvectors; svm_model.models{mIndex}.sv_indices(~logicals)];
    end
    bvectors = unique(bvectors);
    ubvectors = unique(ubvectors);
end

function  visualize(dataset,svm_model,cost)
    numClass = size(dataset,1);
    numSample = cellfun(@length,dataset);
    AC = getActualClass(numSample);
    labels = strcat({'Class '}, num2str((1:numClass)'));
    group = ordinal(AC, labels);
    D = cell2mat(dataset);
    mn = min(D); mx = max(D); n = 1000;
    
    clrLite = [1 0.6 0.6 ; 0.6 1 0.6 ; 0.6 0.6 1; 1 0.6 1];
    clrDark = [0.7 0 0 ; 0 0.7 0 ; 0 0 0.7; 0.7 0 0.7];

    [X, Y] = meshgrid( linspace(mn(1),mx(1),n), linspace(mn(2),mx(2),n) );
    Xl = X(:); Yl = Y(:);
    [gridDC] = classify([Xl Yl], svm_model,numClass);
    [DC] = classify(D, svm_model,numClass);
    
    %plotting all  points
    figure(),set(gcf, 'WindowStyle', 'docked'),hold on;
    image(Xl, Yl, reshape(gridDC, n, n))
    axis xy, box on, colormap(clrLite);
    
    %Superimposing bounded and unbounded
    [bv_indices,ubv_indices] =  computeSupportVectors(svm_model,cost);
  
     gscatter(D(ubv_indices,1), D(ubv_indices,2), group(ubv_indices), clrDark, '.', 5);
     gscatter(D(bv_indices,1), D(bv_indices,2), group(bv_indices), clrDark, '+', 5);
     
%      gscatter(D(:,1), D(:,2), group, clrDark, 'o', 5);
     
    
    %Superimposing wrongly classified points
%     bad = (DC ~= AC);
%     plot(D(bad,1), D(bad,2), 'yx', 'MarkerSize', 10)
    axis([mn(1) mx(1) mn(2) mx(2)]);
    axis tight
    xlabel('Dimension 1'), ylabel('Dimension 2'); 
    hold off 
   end