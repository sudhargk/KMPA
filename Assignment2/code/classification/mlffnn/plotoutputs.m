function plotoutputs(gridX, gridY, gridZ)
    %PLOTOUTPUTS Surface plot of classifier output values for bivariate
    %data
    
    colors = lines();
    numClasses = size(gridZ,3);
    
    inc1 = max(10, size(gridX,1)/100);
    inc2 = max(10, size(gridX,2)/100);
    gridX = gridX(1:inc1:end, 1:inc2:end);
    gridY = gridY(1:inc1:end, 1:inc2:end);
    gridZ = gridZ(1:inc1:end, 1:inc2:end,:);
    figure, set(gcf, 'WindowStyle', 'docked'), hold on
    xlabel('Dimension 1'), ylabel('Dimension 2'), zlabel('Estimated output')
    for cIndex=1:numClasses
        surf(gridX, gridY, gridZ(:,:,cIndex),'FaceColor',colors(cIndex,:));
    end
    hold off;
    axis tight;
    title('Output layer outputs after training'),
    
end

