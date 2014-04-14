%%Plots target output vs model output
% data : specifies the input data type 'univariate','bivariate','multivariate'
% basisType : 'Guassian','Polynomial'
% E.g 
%       plot_4('multivariate',1,3,0);
%       plot_4('bivariate',1,3,0);
%       plot_4('univariate',1,3,0);

function [] =plot_4(dataset,cost,gamma,coef)
    
    %kernel =  'gaussian';
    if(nargin<5)
        dataset = 'bivariate';
        cost = 1; gamma = 3; coef = 0; 
    end
    eps=0.1;
    [trainX,trainT] = importd(dataset,'train');
    [testX,testT] = importd(dataset,'test');
    [valX,valT] = importd(dataset,'val');
    
    [trainX,testX,valX] = normalize(trainX,testX,valX);
%    [svroptions] = buildsvroptions(cost,kernel,a,b,d);
    [svroptions] = buildsvroptions(cost,coef,gamma,eps);
    [model]=c_svr_train(trainX,svroptions,trainT);
    trainET = c_svr_test(trainX,model);
    valET = c_svr_test(valX,model);
    testET = c_svr_test(testX,model);
    
    step = 40;
    step1 = 5;
    step2 = 5;
    
    figure;
    set(gcf, 'WindowStyle', 'docked');
    hold on;
    scatter(trainT(1:step:end),trainET(1:step:end,:),'ro');
    p1=plot(trainT,trainT,'-');
    set(p1,'color','black');
    grid on;
    hold off;
    legend('Training data');
    xlabel('Target Output');
    ylabel('Model Output');
    title(['Target Output vs Model Output - ' dataset])
    axis equal;
    figure;
    set(gcf, 'WindowStyle', 'docked');
    hold on;
    scatter(testT(1:step1:end),testET(1:step1:end,:),'go');
    p2=plot(trainT,trainT,'-');
    set(p2,'color','black');
    grid on;
    hold off;
    legend('Test data');
    xlabel('Target Output');
    ylabel('Model Output');
    title(['Target Output vs Model Output - ' dataset])
    axis equal;
    figure;
    set(gcf, 'WindowStyle', 'docked');
    hold on;
    scatter(valT(1:step2:end),valET(1:step2:end,:),'bo');
    p3=plot(trainT,trainT,'-');
    set(p3,'color','black');
    grid on;
    hold off;
    legend('Validation data');
    xlabel('Target Output');
    ylabel('Model Output');
    title(['Target Output vs Model Output - ' dataset])
    axis equal;
    
   
end
