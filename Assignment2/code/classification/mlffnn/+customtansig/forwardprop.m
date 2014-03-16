function da = forwardprop(dn,n,a,param)
%TANSIG.FORWARDPROP

% Copyright 2012 The MathWorks, Inc.

beta = BETA;
da = bsxfun(@times,dn,beta*(1-(a.*a)));
