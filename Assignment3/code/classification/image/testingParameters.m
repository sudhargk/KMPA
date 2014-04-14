function [] = testingParameters()
     if(nargin<6)
        dataset = 'image';
        kernel =  'gaussian';
        cost = 1; avar = 16:16:160; b = 3; d = 2; 
     end
    path = fullfile(pwd,'..','..','..','data',dataset,'data');
    load(path);    
    valAccuracy = zeros(length(avar),1);
    kGramError = zeros(length(avar),1);
    index = 1;
    for variable = avar
        a = variable;
        [svmoptions,is_custom_kernel] = buildSVMOptions(cost,kernel,a,b,d);
        [~,kGramError(index)]=buildKernelGram(trainset,trainset,kernel,a,b,d);
        [numClass,ntrainset,trainActualClass,nvalset,valActualClass] = initData(trainset,valset,kernel,a,b,d,is_custom_kernel);
        [svm_model] = train(ntrainset,svmoptions,trainActualClass);
        [confusion]=testData(nvalset,svm_model,valActualClass,numClass);       
        [~,valAccuracy(index)]=computeMetrics(confusion,numClass);
        index = index+1;
    end
    figure(1);
    plot(avar,valAccuracy,'b');
    title('Testing parameters');
    xlabel('');ylabel('Validation Accuracy');
    figure(2);
    plot(avar,kGramError,'r');
    title('Testing parameters');
    xlabel('');ylabel('Kernel Gram Error');  
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