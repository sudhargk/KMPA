function [] = generatedata(numPoints,filename)
   sd = 0.3;
   X  = (1:numPoints)' / numPoints;
   T = f(X) + sd*randn(numPoints,1);
   dlmwrite(filename,[X T],'delimiter',' ');
end
