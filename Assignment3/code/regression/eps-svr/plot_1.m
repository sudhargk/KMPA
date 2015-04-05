
%   plot_1(1,20,0,0.5)

function [] =plot_1(cost,gamma,coef,eps)
    
    %kernel =  'gaussian';
    if(nargin<5)
        
        cost = 1; gamma = 20; coef = 0; eps=0.5;
    end
    
    dataset = 'univariate';
    [trainX,trainT] = importd(dataset,'train');
    [testX,testT] = importd(dataset,'test');
    [valX,valT] = importd(dataset,'val');
    step = 20;
    length(trainX)
    [trainX,testX,valX] = normalize(trainX(1:step:end),testX,valX);
    trainT = trainT(1:step:end);
    length(trainX)
%    [svroptions] = buildsvroptions(cost,kernel,a,b,d);
    [svroptions] = buildsvroptions(cost,coef,gamma,eps);
    [model]=c_svr_train(trainX,svroptions,trainT);
    trainET = c_svr_test(trainX,model);
    
    [bvectors,ubvectors] =  computeSupportVectors(model,cost,eps);
    
    
    step1 = 5;
    step2 = 5;
    figure;
    hold on;
    scatter(trainX,trainT,'go');
    p1=plot(trainX,trainET,'-');
    set(p1,'color','black');
    p2=plot(trainX,trainET + eps,'-');
    p3=plot(trainX,trainET - eps,'-');
%     hold off;
%     figure;
%     hold on;
    scatter(trainX(bvectors),trainT(bvectors,:),'co');
%     p1=plot(trainX,trainET,'-');
%     p2=plot(trainX,trainET + eps,'-');
%     p3=plot(trainX,trainET - eps,'-');
%     set(p1,'color','black');
%     hold off;
%     figure;
%     hold on;
    scatter(trainX(ubvectors),trainT(ubvectors,:),'ro');
%     p1=plot(trainX,trainET,'-');
%     p2=plot(trainX,trainET + eps,'-');
%     p3=plot(trainX,trainET - eps,'-');
%     set(p1,'color','black');
    title('Plot of \epsilon - tube, Target output and Approximated function on univariate dataset');
    legend('Training data','f','f + \epsilon','f - \epsilon','bounded','unbounded');
    xlabel('input');
    ylabel('target');
    hold off;
end

function [bvectors,ubvectors] =  computeSupportVectors(svm_model,cost,eps)
     bvectors = [];
     ubvectors = [];
    logicals =  ( abs(svm_model.sv_coef - cost) == 0);
    bvectors = [svm_model.sv_indices(logicals)];
    ubvectors = [svm_model.sv_indices(~logicals)];
    bvectors = unique(bvectors);
    ubvectors = unique(ubvectors);
end
