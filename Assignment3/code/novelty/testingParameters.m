%%  
%Implementation of bayes classifier using gaussian distribution
% @dataset = 'overlapping','linearlySeparable','nonlinearlySeparable'
% @kernel = 'linear', 'polynomial', 'gaussian'
% @cost = C-SVM Cost
% @a,@b,@d = for polynomial (a*x'.y +b)^d  
% @a = for gaussian  exp(-a*|x-v|^2);
% @numCluster  = 1 x C  number of cluster per class
%
function [] = testingParameters()
     if(nargin<6)
        dataset = 'linearlySeparable';
        kernel =  'gaussian';
        costvar = 2.^([-5 5]); avar = 2.^([-5:5]); b = 3; d = 2; 
     end
    path = fullfile(pwd,'..','..','..','data',dataset,'data');
    load(path);    
%     [trainset,testset,valset]=normalize(trainset,testset,valset);
    valAccuracy = zeros(length(avar),length(costvar));
    kGramError = zeros(length(avar),1);
   cindex = 1;
    for cost = costvar
         rindex =1;
        for a = avar
            [svmoptions,~] = buildSVMOptions(cost,kernel,a,b,d);
            [~,kGramError(rindex,cindex)]=buildKernelGram(trainset,trainset,kernel,a,b,d);
             
            [svm_model] = train(trainset,svmoptions);
            [confusion]=testData(valset,svm_model);       
            [~,valAccuracy(rindex,cindex)]=computeMetrics(confusion,numClass);
            rindex=rindex+1;
        end
        cindex = cindex+1;
    end
    figure(1);
    imagesc(mat2gray(valAccuracy));
    title('Testing parameters Validation Accuracy');
    xlabel('Cost C');ylabel('Gaussian inverse width');
    figure(2);
    imagesc(mat2gray(kGramError));
    title('Testing parameters Kernel Gram Error');
    xlabel('Cost C');ylabel('Gaussian inverse width');  
end


function [svmoptions,is_custom_kernel] = buildSVMOptions(cost,kernel,gamma,coef,degree)
    soptions ='-s 0';
%     soptions ='-s 1';
    koptions = '-t';
    is_custom_kernel = false;
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
        case 'histogram'
             koptions = [koptions ' 4'];
             is_custom_kernel = true;
    end
    coptions = ['-c ' num2str(cost)];
%     noptions = ['-n 1'];
    boptions = '-b 1';
    svmoptions = [soptions ' ' koptions ' ' coptions,' ',boptions];
end