function [Cinv, isInv] = buildPriorCovariance(prSpec, prInds, prGrp)
% Use prior spec to build a covariance matrix
%
% [Cinv, isInv] = buildPriorCovariance(prSpec, prInds, prGrp)

nPriors = numel(prSpec);

prGrpInds = grp2idx(prGrp);