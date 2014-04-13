

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
        kernel =  'gaussian';
        cost = 1; a = 0.5; b = 0.6; d = 5; 
     end
    path = fullfile(pwd,'..','..','..','data',dataset,'data');
     
    load(path);    
    buildKernelGram(trainset,trainset,kernel,a,b,d);
    [svmoptions,is_custom_kernel] = buildSVMOptions(cost,kernel,a,b,d);
    [numClass,ntrainset,trainActualClass,ntestset,testActualClass] = initData(trainset,testset,kernel,a,b,d,is_custom_kernel);
    [svm_model] = train(ntrainset,svmoptions,trainActualClass);
    [confusion]=testData(ntestset,svm_model,testActualClass,numClass);       
    [~,overallAcc]=computeMetrics(confusion,numClass);
     format shortg;
     display(confusion);
     display(overallAcc);
       
    visualize(trainset,is_custom_kernel,kernel,a,b,d,svm_model,cost);
   
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



