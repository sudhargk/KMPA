<<<<<<< HEAD
=======

>>>>>>> 1a389ff681b826be7e7528f6b64a5811083343d4
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
<<<<<<< HEAD
end
=======
end
>>>>>>> 1a389ff681b826be7e7528f6b64a5811083343d4
