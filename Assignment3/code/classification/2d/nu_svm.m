
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
function []= nu_svm(dataset,kernel,nu,a,b,d)
    if(nargin<6)
        dataset = 'overlapping';
        kernel =  'gaussian';
        nu = 0.15; a = 0.3; b = 34; d = 4; 
     end
    path = fullfile(pwd,'..','..','..','data',dataset,'data');
     
    load(path);    
    buildKernelGram(trainset,trainset,kernel,a,b,d);
    [svmoptions,is_custom_kernel] = buildSVMOptions(nu,kernel,a,b,d);
    [numClass,ntrainset,trainActualClass,ntestset,testActualClass] = initData(trainset,testset,kernel,a,b,d,is_custom_kernel);
    [svm_model] = train(ntrainset,svmoptions,trainActualClass);
    [confusion]=testData(ntestset,svm_model,testActualClass,numClass);       
    [~,overallAcc]=computeMetrics(confusion,numClass);
     format shortg;
     display(confusion);
     display(overallAcc);
       
    visualize(trainset,is_custom_kernel,kernel,a,b,d,svm_model,1);
      
end
function [svmoptions,is_custom_kernel] = buildSVMOptions(nu,kernel,gamma,coef,degree)
    soptions ='-s 1';
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
            is_custom_kernel = true;        case 'gaussian'
            koptions = [koptions ' 2'];
            koptions = [koptions ' -g ' num2str(gamma)];
    end
    nuoptions = ['-n ' num2str(nu)];
    boptions = '-b 1';
    svmoptions = [soptions ' ' nuoptions ' ' koptions ,' ',boptions];
<<<<<<< HEAD
end
=======
end
>>>>>>> 1a389ff681b826be7e7528f6b64a5811083343d4
