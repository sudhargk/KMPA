%%Plots target output vs model output
% data : specifies the input data type 'univariate','bivariate','multivariate'
% basisType : 'Guassian','Polynomial'
% numbasis : number of basis
% variance : variance parameter for the guassian basis
%%Plots target output and model output vs input feature vector for
%% data using MLFFNN
%  tr - tr output from train phase
%  inputs - inputs to NNET
%  outputs - outputs from NNET

function [] =plot_3(tr,targets, outputs,type)
    input = 1: length(targets);
    trainMask = input.*tr.trainMask{1};trainMask=trainMask(~isnan(trainMask));
    testMask = input.*tr.testMask{1};testMask=testMask(~isnan(testMask));
    valMask = input.*tr.valMask{1};valMask=valMask(~isnan(valMask));
    trainO=outputs(:,trainMask); trainT=targets(:,trainMask);
    testO=outputs(:,testMask); tesT=targets(:,testMask);
    valO=outputs(:,valMask); valT=targets(:,valMask);   
   
   generatePlot(trainT,trainO,type,'training','ro');
   generatePlot(tesT,testO,type,'test','g+');
   generatePlot(valT,valO,type,'validation','b*');
    
end

function [] = generatePlot(Target,Output,type,data,color)
     step = 1;
     figure();
     set(gcf, 'WindowStyle', 'docked');
     scatter(Target(1:step:end),Output(1:step:end),color),hold on;
     plot(Target,Target,'k');
     xlabel('Target Output');
     ylabel('Model Output');
     title(['Scatter plot for Target Output vs Model Output for ' type ' dataset on ' data ' data']);
     axis equal;
     grid on;
     hold off;
end
