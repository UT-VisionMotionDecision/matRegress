function model = autoFitGLM(X,Y, prspec, varargin)



dcTerm = range(X)==0;
nwts   = size(X,2);

if any(dcTerm)
	DCflag = true;
end

if numel(prspec)==1

	buildPriorCovariance(prspec, {inds}, {prgrp})