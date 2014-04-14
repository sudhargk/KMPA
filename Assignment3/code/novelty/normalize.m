function [trainset,testset,valset] = normalize(trainset,testset,valset)
    numClass = length(trainset);
    for index = 1:numClass
        train = trainset{index}; test = testset{index}; val = valset{index};
        T =[train;test;val];
        MIN = min(T);
        MAX = max(T);
        train = bsxfun(@minus,train,MIN);
        train = bsxfun(@rdivide,train,MAX-MIN);
        test = bsxfun(@minus,test,MIN);
        test = bsxfun(@rdivide,test,MAX-MIN);
        val = bsxfun(@minus,val,MIN);
        val = bsxfun(@rdivide,val,MAX-MIN);
        trainset{index}=train; valset{index}=val;testset{index}=test;
    end
end