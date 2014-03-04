%%Plots target output and model output vs input feature vector
% data : specifies the input data type 'univariate','bivariate'
% basisType : 'Guassian','Polynomial'
% numbasis : number of basis
% variance : variance parameter for the guassian basis
% lambda : regularization parameter
% E.g 
%       plot_4('bivariate','Guassian',10,exp(-10),2);
%       plot_4('univariate','Polynomial',10,exp(-10),1);

function [] =plot_4(data,basisType,basis,lambda,variance)
    [trainX,trainT] = importd(data,'train'); 
    [testX,testT] = importd(data,'test');
    [valX,valT] = importd(data,'val');
    [trainX,testX,valX] = normalize(trainX,testX,valX);
    [trainXPhi,M] = computeDesignMatrix(trainX,basisType,basis,variance);
    testXPhi = computePhiFromDesign(testX,basisType,basis,variance,M);
    valXPhi = computePhiFromDesign(valX,basisType,basis,variance,M);
    W = train(trainXPhi,trainT,lambda);
    generatePlot(trainX,trainXPhi*W,trainT,data,'train');
    generatePlot(testX,testXPhi*W,testT,data,'test');
    generatePlot(valX,valXPhi*W,valT,data,'validation');
    
end

function [] = generatePlot(input, output, target, data,dataType)
    figure();
    switch (data);
        case 'univariate' 
                    X = input(:,1); Y2 = output(:,1); Y1 = target(:,1); 
                    Z = zeros(size(input,1),1);
        case 'bivariate'
                    X = input(:,1); Y2 = output(:,1); Y1 = target(:,1);
                    Z = input(:,2);
    end
    hold on;
    title(['Plot of Target output and Model output on ' dataType ' data for ' data ' dataset'])
    
    plot3(X,Y1,Z,'g+');
    plot3(X,Y2,Z,'ro');
    legend('Target Output','Model Output');
    xlabel('Dimension 1');
    ylabel('Output');
    zlabel('Dimension 2');
    hold off;
end
