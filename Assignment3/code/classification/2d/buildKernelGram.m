%%
%   Builds the Kernel Gram Matrix for different kernels
%   @kernel = 'linear', 'polynomial', 'gaussian'
%   @flag = compute error if true else not
%   @a,@b,@d = for polynomial (a*x'.y +b)^d  
%   @a = for gaussian  exp(-a*|x-v|^2);

function[G,error]= buildKernelGram(X1,X2,kernel,a,b,d)
    numClass = size(X1,1);
    numX1Sample = cellfun('size',X1,1);
    X1Samples = sum(numX1Sample);
    
    numX2Sample = cellfun('size',X2,1);
    X2Samples = sum(numX2Sample);
    
    X1 = cell2mat(X1);
    X2 = cell2mat(X2);
    
    handle = str2func(kernel);
    [G] = handle(X1,X2,a,b,d);
    G = mat2gray(G);
    ideal = zeros(X1Samples,X2Samples);
    
    prev1 = 1;prev2 = 1;
    for cIndex = 1:numClass
        ssize1 = numX1Sample(cIndex);
        ssize2 = numX2Sample(cIndex);
        ideal(prev1:ssize1,prev2:ssize2)=1;
        prev1 = ssize1+1; prev2 = ssize2+1;
    end
    error = norm(ideal-G,'fro');
%     colormap gray;
%     imagesc(G);
    G = [(1:X1Samples)' G];
end

function [G] = linear(X1,X2,a,b,d)
    G = X1 * X2';
end
function [G] = polynomial(X1,X2,a,b,d)
    G = (a*X1*X2'+ b).^d;
    tX1 = sqrt((a*sum(X1.*X1,2)+ b).^d);
    tX2 = sqrt((a*sum(X2.*X2,2)+ b).^d);
    G = bsxfun(@rdivide,G,tX1);
    G = bsxfun(@rdivide,G,tX2');
end

function [G] = gaussian(X1,X2,a,b,d)
    G = pdist2(X1,X2);
    G = exp(-a*G.^2);
end
function [G] = histogram(X1,X2,a,b,d)
    G = pdist2(X1,X2,@histointersection);
end

function [dist] = histointersection(X,Y)
        dist = bsxfun(@min,Y,X);
        dist=sum(dist,2);
end
