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
        dataset = 'nonlinearlySeparable';
        kernel =  'gaussian';
        costvar=1; a = 0.6; b = 0.6; d = 3; 
     end
    path = fullfile(pwd,'..','..','..','data',dataset,'data');
    load(path);    
% %     [trainset,testset,valset]=normalize(trainset,testset,valset);

%     kGramError = zeros(length(var),1);
%     for rindex =1:length(var);
% %         [svmoptions,~] = buildSVMOptions(cost,kernel,a,b,var(rindex));
% %         [~,kGramError(rindex)]=buildKernelGram(trainset,trainset,kernel,a,b,var(rindex));
% %          [svmoptions,~] = buildSVMOptions(cost,kernel,a,var(rindex),d);
% %         [~,kGramError(rindex)]=buildKernelGram(trainset,trainset,kernel,a,var(rindex),d);
%             [svmoptions,~] = buildSVMOptions(cost,kernel,var(rindex),b,d);
%         [~,kGramError(rindex)]=buildKernelGram(trainset,trainset,kernel,var(rindex),b,d);
% 
%     end
%    figure(1);
%    plot(var,kGramError,'r');
%    title('Testing parameters Kernel Gram Error');
%    xlabel('Gaussian Inverse Width');ylabel('Kernel gram error');  
   
    valAccuracy = zeros(length(costvar),1);    
   for cindex = 1:length(costvar)
        cost = costvar(cindex);
        [svmoptions,is_custom_kernel] = buildSVMOptions(cost,kernel,a,b,d);
        [numClass,ntrainset,trainActualClass,ntestset,testActualClass] = initData(trainset,testset,kernel,a,b,d,is_custom_kernel);
        [svm_model] = train(ntrainset,svmoptions,trainActualClass);
        [confusion]=testData(ntestset,svm_model,testActualClass,numClass);       
        [~,valAccuracy(cindex)]=computeMetrics(confusion,numClass);
    end
    figure(2);
    plot(costvar,valAccuracy,'r');
    title('Testing parameters Validation Accuracy');
    ylabel('Accuracy'); xlabel('Cost C');
   
   
end


function [svmoptions,is_custom_kernel] = buildSVMOptions(cost,kernel,gamma,coef,degree)
    soptions ='-s 0';
    koptions = '-t';
    is_custom_kernel = false;
    switch(kernel)
        case 'linear'  
            koptions = [koptions ' 0'];
        case 'polynomial'
%             koptions = [koptions ' 1'];
%             koptions = [koptions ' -g ' num2str(gamma)];
%             koptions = [koptions ' -r ' num2str(coef)];
%             koptions = [koptions ' -d ' num2str(degree)];
            koptions = [koptions ' 4'];
            is_custom_kernel = true;
        case 'gaussian'
            koptions = [koptions ' 2'];
            koptions = [koptions ' -g ' num2str(gamma)];
    end
    coptions = ['-c ' num2str(cost)];
    svmoptions = [soptions ' ' koptions ' ' coptions];
end
