%%Plots target output vs model output
% data : specifies the input data type 'univariate','bivariate','multivariate'
% basisType : 'Guassian','Polynomial'
% numbasis : number of basis
% variance : variance parameter for the guassian basis
% lambda : regularization parameter
% E.g 
%       plot_5('multivariate','Gaussian',10,exp(-10),2);
%       plot_5('bivariate','Gaussian',10,exp(-10),2);
%       plot_5('univariate','Polynomial',10,exp(-10),1);

function [] =plot_5(data,basisType,basis,lambda,variance)
    [trainX,trainT] = importd(data,'train');
    [testX,testT] = importd(data,'test');
    [valX,valT] = importd(data,'val');
    [trainX,testX,valX]=normalize(trainX,testX,valX);
    
    [M,~] = computeClusterMeans(trainX,basis);
    trainXPhi = computeDesignMatrix(trainX,basisType,basis,variance,M);
    testXPhi = computeDesignMatrix(testX,basisType,basis,variance,M);
    valXPhi = computeDesignMatrix(valX,basisType,basis,variance,M);
    
    W = train(trainXPhi,trainT,lambda);
    hold on;
    step = 1;
    scatter(trainT(1:step:end),trainXPhi(1:step:end,:)*W,'ro');
    scatter(testT(1:step:end),testXPhi(1:step:end,:)*W,'g+');
    scatter(valT(1:step:end),valXPhi(1:step:end,:)*W,'b*');
    ezplot('x');
    legend('train','test','validation','Location', 'NW');
    xlabel('Target Output');
    ylabel('Model Output');
    title(['Target Output vs Model Output - ' data])
%     T = [trainT;testT;valT];
%     axis([min(T) max(T) min(T) max(T)]);
    axis equal;
    
    grid on;
    hold off; 
end
