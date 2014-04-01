%%Plots target output vs model output
% data : specifies the input data type 'univariate','bivariate','multivariate'
% basisType : 'Guassian','Polynomial'
% numbasis : number of basis
% variance : variance parameter for the guassian basis
% lambda : regularization parameter
% E.g 
%       plot_3('multivariate',10,exp(-10));
%       plot_3('bivariate',10,exp(-10));
%       plot_3('univariate',10,exp(-10));

function [] =plot_3(data,basis,lambda)
    basisType = 'Gaussian';
    [trainX,trainT] = importd(data,'train');
    [testX,testT] = importd(data,'test');
    [valX,valT] = importd(data,'val');
    [trainX,testX,valX]=normalize(trainX,testX,valX);
    
    [M,tichonovDist,width] = computeClusterMeans(trainX(1:10),basis);
    [trainXPhi] = computeDesignMatrix(trainX,basisType,basis,M,width);
    testXPhi = computeDesignMatrix(testX,basisType,basis,M,width);
    valXPhi = computeDesignMatrix(valX,basisType,basis,M,width);
    W = train(trainXPhi,trainT,lambda,tichonovDist);
    hold on;
    step = 10;

    scatter(trainT(1:step:end),trainXPhi(1:step:end,:)*W,'ro');
    scatter(testT(1:step:end),testXPhi(1:step:end,:)*W,'g+');
    scatter(valT(1:step:end),valXPhi(1:step:end,:)*W,'b*');
    plot(trainT,trainT,'k.');
    legend('train','test','validation','Location', 'NW');
    xlabel('Target Output');
    ylabel('Model Output');
    title(['Target Output vs Model Output - ' data])
    axis equal;
    
    grid on;
    hold off; 
end

