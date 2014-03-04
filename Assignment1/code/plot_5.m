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
    figure,
    set(gcf, 'WindowStyle', 'docked');
    hold on;
<<<<<<< HEAD
    step = 4;
    scatter(trainT(1:step:end),trainXPhi(1:step:end,:)*W,'ro');
    scatter(testT(1:step:end),testXPhi(1:step:end,:)*W,'g+');
    scatter(valT(1:step:end),valXPhi(1:step:end,:)*W,'b*');
    plot(trainT, trainT, 'k');
=======
    step = 10;
%     ezplot('x');
    scatter(trainT(1:step:end),trainXPhi(1:step:end,:)*W,'ro');
    scatter(testT(1:step:end),testXPhi(1:step:end,:)*W,'g+');
    scatter(valT(1:step:end),valXPhi(1:step:end,:)*W,'b*');
    plot(trainT,trainT,'k.');
>>>>>>> 77cca7424c1d03cd09657342b67db729b05e9345
    legend('train','test','validation','Location', 'NW');
    xlabel('Target Output');
    ylabel('Model Output');
    title(['Target Output vs Model Output - ' data])
%     T = [trainT;testT;valT];
%     axis([min(T) max(T) min(T) max(T)]);
    axis square; 
    grid on;
    hold off; 
end

