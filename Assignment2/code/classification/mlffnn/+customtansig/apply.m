function a = apply(n,param)

% Copyright 2012 The MathWorks, Inc.

beta = BETA;
a = 2 ./ (1 + exp(-2*beta*n)) - 1;

