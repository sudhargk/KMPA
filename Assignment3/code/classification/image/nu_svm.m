

%%
%Implementation of bayes classifier using gaussian distribution
% @dataset = 'overlapping','linearlySeparable','nonlinearlySeparable'
% @kernel = 'image'
% @cost = C-SVM Cost
% @nu = NU value
% @a,@b,@d = for polynomial (a*x'.y +b)^d  
% @a = for gaussian  exp(-a*|x-v|^2);
% @numCluster  = 1 x C  number of cluster per class
%
function []= nu_svm(dataset,kernel,nu,a,b,d)
     if(nargin<6)
        dataset = 'image';
        kernel =  'gaussian';
        nu=0.27;  a = 98; b = 3; d = 2; 
     end
    path = fullfile(pwd,'..','..','..','data',dataset,'data');
    
    load(path);    
   
    [svmoptions,is_custom_kernel] = buildSVMOptions(nu,kernel,a,b,d);
    [numClass,trainset,trainActualClass,testset,testActualClass] = initData(trainset,testset,kernel,a,b,d,is_custom_kernel);
    [svm_model] = train(trainset,svmoptions,trainActualClass);
    [confusion]=testData(testset,svm_model,testActualClass,numClass,classes);       
    
    [perClassInfo,overallAcc]=computeMetrics(confusion,numClass);
     format shortg;
     display(confusion);
     display(overallAcc);
      
end
function [svmoptions,is_custom_kernel] = buildSVMOptions(nu,kernel,gamma,coef,degree)
    soptions ='-s 1';
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
    nuoptions = ['-n ' num2str(nu)];
    svmoptions = [soptions ' ' nuoptions ' ' koptions ];
end





