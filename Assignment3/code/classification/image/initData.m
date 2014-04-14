function [numClass,ntrainset,trainActualClass,ntestset,testActualClass] = initData(trainset,testset,kernel,a,b,d,is_custom_kernel)
    numClass = size(trainset,1);
    numSample = cellfun('size',trainset,1);
    trainActualClass = getActualClass(numSample);
   
    numSample = cellfun('size',testset,1);
    testActualClass = getActualClass(numSample);
    
    if(is_custom_kernel)
      ntrainset = buildKernelGram(trainset,trainset,kernel,a,b,d);
      ntestset = buildKernelGram(testset,trainset,kernel,a,b,d);
    else
       ntrainset = cell2mat(trainset);     
       ntestset = cell2mat(testset);
    end
end
