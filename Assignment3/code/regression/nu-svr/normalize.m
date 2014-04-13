function [trainX,testX,valX] = normalize(trainX,testX,valX)
    T =[trainX;testX;valX];
    MIN = min(T);
    MAX = max(T);
    trainX = bsxfun(@minus,trainX,MIN);
    trainX = bsxfun(@rdivide,trainX,MAX-MIN);
    testX = bsxfun(@minus,testX,MIN);
    testX = bsxfun(@rdivide,testX,MAX-MIN);
    valX = bsxfun(@minus,valX,MIN);
    valX = bsxfun(@rdivide,valX,MAX-MIN);
end
