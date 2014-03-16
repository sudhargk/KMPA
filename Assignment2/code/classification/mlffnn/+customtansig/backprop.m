function dn = backprop(da,n,a,param)
%TANSIG.BACKPROP

% Copyright 2012 The MathWorks, Inc.

beta = BETA;
dn = bsxfun(@times,da,beta*(1-(a.*a)));
