function visualize(dataset,is_custom_kernel,kernel,a,b,d,svm_model,cost)
        numClass = size(dataset,1);
        tempset = cell(2,1);
        idx = 1:numClass;
        if(~strcmp(kernel,'polynomial'))
            for index = idx
                tempset{1} = cell2mat(dataset(index));
                tempset{2} = cell2mat(dataset(idx~=index));
                view(tempset,is_custom_kernel,kernel,a,b,d,svm_model.models{index},cost,false);
            end
        end
        view(dataset,is_custom_kernel,kernel,a,b,d,svm_model,cost,true);
end

function  view(dataset,is_custom_kernel,kernel,a,b,d,svm_model,cost,flag)
    numClass = size(dataset,1);
    numSample = cellfun(@length,dataset);
    AC = getActualClass(numSample);
    labels = strcat({'Class '}, num2str((1:numClass)'));
    group = ordinal(AC, labels);
    D = cell2mat(dataset);
    mn = min(D); mx = max(D); n = 1000;
    
    clrLite = [1 0.6 0.6 ; 0.6 1 0.6 ; 0.6 0.6 1; 1 0.6 1];
    clrDark = [0.7 0 0 ; 0 0.7 0 ; 0 0 0.7; 0.7 0 0.7];
   
    D = cell2mat(dataset);
    mn = min(D); mx = max(D); n = 300;
    [X, Y] = meshgrid( linspace(mn(1),mx(1),n), linspace(mn(2),mx(2),n) );
    Xl = X(:); Yl = Y(:);
    
    if(is_custom_kernel)
        grid = cell(1,1);
        grid{1}=[Xl Yl];
        grid = buildKernelGram(grid,dataset,kernel,a,b,d);
        [gridDC] = classify(grid, svm_model,numClass,flag);
        DDash = buildKernelGram(dataset,dataset,kernel,a,b,d);
        [DC] = classify(DDash, svm_model,numClass,flag);
    else
        [gridDC] = classify([Xl Yl], svm_model,numClass,flag);
        [DC] = classify(D, svm_model,numClass,flag);
    end
    
    %plotting all  points
    figure(),set(gcf, 'WindowStyle', 'docked'),hold on;
    image(Xl, Yl, reshape(gridDC, n, n))
    axis xy, box on, colormap(clrLite);
    
    %Superimposing bounded and unbounded
    [bv_indices,ubv_indices] =  computeSupportVectors(svm_model,cost,flag);
     if(flag)
        gscatter(D(ubv_indices,1), D(ubv_indices,2), group(ubv_indices), clrDark, 'o', 10);
        gscatter(D(bv_indices,1), D(bv_indices,2), group(bv_indices), clrDark, 'v', 10);
     else
        plot(ubv_indices(:,1), ubv_indices(:,2),'ko');
        plot(bv_indices(:,1), bv_indices(:,2), 'kv'); 
     end
     
     
    gscatter(D(:,1), D(:,2), group, clrDark, '.', 6);
     
    axis([mn(1) mx(1) mn(2) mx(2)]);
    axis tight
    xlabel('Dimension 1'), ylabel('Dimension 2'); 
    hold off 
end
   
function [decidedClass]= classify(dataset,svm_model,numClass,flag)
    totalSample = size(dataset,1);
    testingLabels  =  floor((1:totalSample)/ceil(totalSample/numClass))+1;
    testingLabels = testingLabels';    
    if(flag)
        [decidedClass] = ovrpredict(testingLabels,dataset,svm_model); 
    else
        [decidedClass] = svmpredict(testingLabels,dataset,svm_model); 
        decidedClass = decidedClass+1;
    end
    decidedClass = decidedClass';
end

function [bvectors,ubvectors] =  computeSupportVectors(svm_model,cost,flag)
     if(flag)
        bvectors = [];
        ubvectors = [];
        numModel = length(svm_model.models);
        for mIndex = 1:numModel
            logicals =  (cost-abs(svm_model.models{mIndex}.sv_coef)<eps);
            bvectors = [bvectors; svm_model.models{mIndex}.sv_indices(logicals)];
            ubvectors = [ubvectors; svm_model.models{mIndex}.sv_indices(~logicals)];
        end
        bvectors = unique(bvectors);
        ubvectors = unique(ubvectors);
     else
          logicals =  (cost-abs(svm_model.sv_coef)<eps);
          svs = full(svm_model.SVs);
          bvectors = svs(logicals,:);
         ubvectors =  svs(~logicals,:);;
     end
   
end
