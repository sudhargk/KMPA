function plotdecisionregions(inputs, targets, outputs, gridX, gridY, gridZ)
    %PLOTDECISIONREGIONS Decision region plot over a 2D space (Upto 4 classes)
    %Scatter plot of the data superimposed on the decision regions of
    %different classess.
    
    clrLite = [1 0.6 0.6 ; 0.6 1 0.6 ; 0.6 0.6 1; 1 0.6 1];
    clrDark = [0.7 0 0 ; 0 0.7 0 ; 0 0 0.7; 0.7 0 0.7];
    D = inputs';
    [~, DC] = max(outputs);
    [~, gridDC] = max(gridZ, [], 3);
    AC = vec2ind(targets);
    Xl = gridX(:)'; Yl = gridY(:)';
    minimum = min(D);
    maximum = max(D);
    
    %Background image showing class regions
    hold on
    image(Xl, Yl, gridDC)
    colormap(clrLite);
    %Superimposing data points
    gscatter(D(:,1), D(:,2), AC', clrDark, '.', 15);
    bad = (DC ~= AC);
    plot(D(bad,1), D(bad,2), 'yx', 'MarkerSize', 10);
    hold off;
    xlim([minimum(1) maximum(1)]);
    ylim([minimum(2) maximum(2)]);
    title('Decision regions in input space'),
    xlabel('Dimension 1'), ylabel('Dimension 2');
end
