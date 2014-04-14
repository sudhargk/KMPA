

%%
%Implementation of bayes classifier using gaussian distribution
% @dataset = 'overlapping','linearlySeparable','nonlinearlySeparable'
% @kernel = 'linear', 'polynomial', 'gaussian'
% @cost = C-SVM Cost
% @nu = NU value
% @a,@b,@d = for polynomial (a*x'.y +b)^d  
% @a = for gaussian  exp(-a*|x-v|^2);
% @numCluster  = 1 x C  number of cluster per class
%
function []= nu_svm(dataset,kernel,nu,cost,a,b,d)
     if(nargin<6)
        dataset = 'linearlySeparable';
        kernel =  'linear';
        nu=0.6; cost = 1; a = 4; b = 3; d = 2; 
     end
    path = fullfile(pwd,'..','..','..','data',dataset,'data');
    
    load(path);    
   
    svmoptions = buildSVMOptions(cost,nu,kernel,a,b,d);
    buildKernelGram(trainset,trainset,kernel,a,b,d);
    [svm_model] = train(trainset,svmoptions);
    [confusion]=testData(testset,svm_model);       
    [perClassInfo,overallAcc]=computeMetrics(confusion,numClass);
     format shortg;
     display(confusion);
     display(overallAcc);
      visualize(trainset,svm_model,cost);
      
end
function [svmoptions] = buildSVMOptions(cost,nu,kernel,gamma,coef,degree)
    soptions ='-s 1';
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
    nuoptions = ['-n ' num2str(nu)];
    boptions = '-b 1';
    svmoptions = [soptions ' ' nuoptions ' ' koptions ' ' coptions,' ',boptions];
end