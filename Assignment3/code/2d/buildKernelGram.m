%%
%   Builds the Kernel Gram Matrix for different kernels
%   @kernel = 'linear', 'polynomial', 'gaussian'
%   @a,@b,@d = for polynomial (a*x'.y +b)^d  
%   @a = for gaussian  exp(-a*|x-v|^2);

function[G]= buildKernelGram(X1,X2,kernel,a,b,d)
    handle = str2func(kernel);
    [G] = handle(X1,X2,a,b,d);
%     colormap gray;
    imagesc(G);
end

function [G] = linear(X1,X2,a,b,d)
    G = X1 * X2';
end
function [G] = polynomial(X1,X2,a,b,d)
    G = (a*X1*X2'+ b).^d;
end

function [G] = gaussian(X1,X2,a,b,d)
    G = pdist2(X1,X2);
    G = exp(-a*G.^2);
end

