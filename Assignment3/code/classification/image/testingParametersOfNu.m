function [] = testingParametersOfNu()
    if(nargin<6)
        dataset = 'image';
        kernel =  'gaussian';
        nuvar=0.01:0.02:0.3; a = 98; b = 34; d = 4; 
     end
    path = fullfile(pwd,'..','..','..','data',dataset,'data');
    load(path);    
%     [trainset,testset,valset]=normalize(trainset,testset,valset);
% 
%     kGramError = zeros(length(var),1);
%     for rindex =1:length(var);
% %         [svmoptions,~] = buildSVMOptions(nu,kernel,a,b,var(rindex));
% %         [~,kGramError(rindex)]=buildKernelGram(trainset,trainset,kernel,a,b,var(rindex));
% %          [svmoptions,~] = buildSVMOptions(nu,kernel,a,var(rindex),d);
% %         [~,kGramError(rindex)]=buildKernelGram(trainset,trainset,kernel,a,var(rindex),d);
%             [svmoptions,~] = buildSVMOptions(nu,kernel,var(rindex),b,d);
%         [~,kGramError(rindex)]=buildKernelGram(trainset,trainset,kernel,var(rindex),b,d);
% 
%     end
%    figure(1);
%    plot(var,kGramError,'r');
%    title('Testing parameters Kernel Gram Error');
%    xlabel('Coefficient of Polynomial');ylabel('Kernel gram error');  
   
    valAccuracy = zeros(length(nuvar),1);    
   for cindex = 1:length(nuvar)
        nu = nuvar(cindex);
        [svmoptions,is_custom_kernel] = buildSVMOptions(nu,kernel,a,b,d);
        [numClass,ntrainset,trainActualClass,ntestset,testActualClass] = initData(trainset,testset,kernel,a,b,d,is_custom_kernel);
        [svm_model] = train(ntrainset,svmoptions,trainActualClass);
        [confusion]=testData(ntestset,svm_model,testActualClass,numClass,classes);       
        [~,valAccuracy(cindex)]=computeMetrics(confusion,numClass);
    end
    figure(2);
    plot(nuvar,valAccuracy,'r');
    title('Testing parameters Validation Accuracy');
    ylabel('Accuracy'); xlabel('Nu');
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