%%Plots target output and model output vs input feature vector
% data : specifies the input data type 'univariate','bivariate'
% basisType : 'Guassian','Polynomial'
% numbasis : number of basis
% variance : variance parameter for the guassian basis
% lambda : regularization parameter
% E.g 
%       plot_4('bivariate','Guassian',10,exp(-10),2);
%       plot_5('univariate','Polynomial',10,exp(-10),1);

function [] =plot_4(data,basisType,basis,lambda,variance)
    [trainX,trainT] = importd(data,'train'); 
    [testX,testT] = importd(data,'test');
    [valX,valT] = importd(data,'val');
    [trainX,testX,valX] = normalize(trainX,testX,valX);
    [trainXPhi,M] = computeDesignMatrix(trainX,basisType,basis,variance);
    testXPhi = computePhiFromDesign(testX,basisType,basis,variance,M);
    valXPhi = computePhiFromDesign(valX,basisType,basis,variance,M);
    W = train(trainXPhi,trainT,lambda);
    hold on;
    figure(1);
    X = trainX(:,1); Y = trainX(:,2); Z = 0;
    plot3(trainT,trainXPhi*W,'ro');
    scatter(testT,testXPhi*W,'g+');
    scatter(valT,valXPhi*W,'b*');
    axis equal;
    legend('train','test','validation');
    T = [ trainT;testT;valT];
    axis([min(T) max(T) min(T) max(T)]);
    
    grid on;
    hold off; 
end

function [] = generatePlot(input, output, target, modelType)
    switch modelType
      case 'univariate':
    end
end
