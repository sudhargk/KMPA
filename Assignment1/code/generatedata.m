function [] = generatedata(numPoints,filename)
   sd = 0.3;
   X  = linspace(0,1,numPoints)';
   T = f(X) + sd*randn(numPoints,1);
   dlmwrite(filename,[X T],'delimiter',' ');
end
