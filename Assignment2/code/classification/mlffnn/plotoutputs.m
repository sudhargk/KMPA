function plotoutputs(gridX, gridY, gridZ)
    %PLOTOUTPUTS Surface plot of classifier output values for bivariate
    %data (Upto 64 classes)
    
    colors = lines();
    numClasses = size(gridZ,3);
    
    hold on
    xlabel('Dimension 1'), ylabel('Dimension 2'), zlabel('Estimated output')
    for cIndex=1:numClasses
        surf(gridX, gridY, gridZ(:,:,cIndex),'FaceColor',colors(cIndex,:));
    end
    hold off;
    axis tight;
    title('Output surfaces after training'),
    
end

