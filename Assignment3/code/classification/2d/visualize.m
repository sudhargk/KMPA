

function  visualize(dataset,svm_model,cost)
    numClass = size(dataset,1);
    numSample = cellfun(@length,dataset);
    AC = getActualClass(numSample);
    labels = strcat({'Class '}, num2str((1:numClass)'));
    group = ordinal(AC, labels);
    D = cell2mat(dataset);
    mn = min(D); mx = max(D); n = 1000;
    
    clrLite = [1 0.6 0.6 ; 0.6 1 0.6 ; 0.6 0.6 1; 1 0.6 1];
    clrDark = [0.7 0 0 ; 0 0.7 0 ; 0 0 0.7; 0.7 0 0.7];

    [X, Y] = meshgrid( linspace(mn(1),mx(1),n), linspace(mn(2),mx(2),n) );
    Xl = X(:); Yl = Y(:);
    [gridDC] = classify([Xl Yl], svm_model,numClass);
    [DC] = classify(D, svm_model,numClass);
    
    %plotting all  points
    figure(),set(gcf, 'WindowStyle', 'docked'),hold on;
    image(Xl, Yl, reshape(gridDC, n, n))
    axis xy, box on, colormap(clrLite);
    
    %Superimposing bounded and unbounded
    [bv_indices,ubv_indices] =  computeSupportVectors(svm_model,cost);
  
     gscatter(D(ubv_indices,1), D(ubv_indices,2), group(ubv_indices), clrDark, '.', 5);
     gscatter(D(bv_indices,1), D(bv_indices,2), group(bv_indices), clrDark, '+', 5);
     
      gscatter(D(:,1), D(:,2), group, clrDark, 'o', 5);
     
    
    %Superimposing wrongly classified points
%     bad = (DC ~= AC);
%     plot(D(bad,1), D(bad,2), 'yx', 'MarkerSize', 10)
    axis([mn(1) mx(1) mn(2) mx(2)]);
    axis tight
    xlabel('Dimension 1'), ylabel('Dimension 2'); 
    hold off 
end
   
function [bvectors,ubvectors] =  computeSupportVectors(svm_model,cost)
    numModel = length(svm_model.models);
     bvectors = [];
     ubvectors = [];
    for mIndex = 1:numModel
        logicals =  ((svm_model.models{mIndex}.sv_coef-cost)<eps);
        bvectors = [bvectors; svm_model.models{mIndex}.sv_indices(logicals)];
        ubvectors = [ubvectors; svm_model.models{mIndex}.sv_indices(~logicals)];
    end
    bvectors = unique(bvectors);
    ubvectors = unique(ubvectors);
end
