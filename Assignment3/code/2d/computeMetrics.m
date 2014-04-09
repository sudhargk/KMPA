    function [perClassInfo,overallAcc] = computeMetrics(confusion,numClasses)
    x = 1:numClasses;
    perClassInfo = zeros(numClasses, 16);
    for cIndex=1:numClasses
        ind = (x ~= cIndex);
        classRow = confusion(cIndex, :);
        otherRow = sum(confusion(ind, :),1);
        tmp = [classRow; otherRow];
        classCol = tmp(:, cIndex);
        otherCol = sum(tmp(:, ind),2);
        TP = classCol(1); FN = otherCol(1);
        FP = classCol(2); TN = otherCol(2);
        % First 8 columns of perClassInfo :-
        % P=TP+FN, N=TN+FP, P'= TP+FP, N'=TN+FN, TP, TN, FP, FN
        perClassInfo(cIndex, 1:8) = [TP+FN FP+TN TP+FP TN+FN TP TN FP FN];
    end
    %  9 - TPR = TP/P
    perClassInfo(:, 9)  = perClassInfo(:, 5) ./ perClassInfo(:, 1);
    % 10 - FPR = FP/N
    perClassInfo(:, 10) = perClassInfo(:, 7) ./ perClassInfo(:, 2);
    % 11 - ACC = (TP+TN)/(P+N)
    perClassInfo(:, 11) = (perClassInfo(:, 5) + perClassInfo(:, 6)) ./ (perClassInfo(:, 1) + perClassInfo(:, 2));
    % 12 - SPC = TN/N = 1-FPR
    perClassInfo(:, 12) = 1 - perClassInfo(:, 10);
    % 13 - PPV = TP/P'
    perClassInfo(:, 13) = perClassInfo(:, 5) ./ perClassInfo(:, 3);
    % 14 - NPV = TN/N'
    perClassInfo(:, 14) = perClassInfo(:, 6) ./ perClassInfo(:, 4);
    % 15 - FDR = FP/P' = 1 - PPV
    perClassInfo(:, 15) = 1 - perClassInfo(:, 13);
    % 16 - MCC = (TP*TN - FP*FN) / sqrt(P*N*P'*N')
    den = perClassInfo(:, 1) .* perClassInfo(:, 2) .* perClassInfo(:, 3) .* perClassInfo(:, 4);
    num = (perClassInfo(:, 5) .* perClassInfo(:, 6)) - (perClassInfo(:, 7) .* perClassInfo(:, 8));
    ind = (den ~= 0);
    perClassInfo(ind, 16) = num(ind, :) ./ sqrt(den(ind, :));
    overallAcc = sum(diag(confusion)) ./ sum(sum(confusion));
end
