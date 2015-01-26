function [Cinv, isInv] = buildPriorCovariance(prSpec, prInds, prGrp)
% Use prior spec to build a covariance matrix
%
% [Cinv, isInv] = buildPriorCovariance(prSpec, prInds, prGrp)
% prSpec is a prior "object" (it's a struct array)
% prSpec
% 		.label     - name of this prior (eg. 'Ridge1')
%		.fun   	   - function to generate the prior covariance (eg. @ridge)
% 		.dflthyprs - default hyperparameters (eg. .1)
% 		.nhyprs    - number of hyperparameters required
% 		.isInv     - boolean 
% prInds - cell-array
% prGrp  - which prior to use for each of prInds		

nPriors = numel(prSpec);

prGrpInds = grp2idx(prGrp);