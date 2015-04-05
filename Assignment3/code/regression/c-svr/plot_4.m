%%Plots target output vs model output
% data : specifies the input data type 'univariate','bivariate','multivariate'
% basisType : 'Guassian','Polynomial'
% E.g 
%       plot_3('multivariate',10,exp(-10));
%       plot_3('bivariate',10,exp(-10));
%       plot_3('univariate',10,exp(-10));

function [] =plot_4(dataset,kernel,cost,a,b,d)
     if(nargin<6)
        dataset = 'bivariate';
        kernel =  'gaussian';
        cost = 1; a = 3; b = 3; d = 2; 
    end
    [trainX,trainT] = importd(dataset,'train');
    [testX,testT] = importd(dataset,'test');
    [valX,valT] = importd(dataset,'val');
    
    [trainX,testX,valX] = normalize(trainX,testX,valX);
    [svroptions] = buildsvroptions(cost,kernel,a,b,d);
    [model]=c_svr_train(trainX,svroptions,trainT);
    trainET = c_svr_test(trainX,model);
    valET = c_svr_test(valX,model);
    testET = c_svr_test(testX,model);
    
    step = 10;
    
    scatter(trainT(1:step:end),trainET(1:step:end,:),'ro');
    scatter(testT(1:step:end),testET(1:step:end,:),'g+');
    scatter(valT(1:step:end),valET(1:step:end,:),'b*');
    plot(trainT,trainT,'k.');
    legend('train','test','validation');
    xlabel('Target Output');
    ylabel('Model Output');
    title(['Target Output vs Model Output - ' dataset])
    axis equal;
    
    grid on;
    hold off; 
end

