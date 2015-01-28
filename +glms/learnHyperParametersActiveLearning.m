function S = learnHyperParametersActiveLearning(X,Y,distr, prspec, prior_inds, prior_grp)
% SomeOutput = learnHyperParametersActiveLearning(X,Y,prspec, prior_inds, prior_grp)
import gpao.*
maxIter = 50;
domain = reshape([prspec(:).hyprsRnge], 2, [])';
d = size(domain,1);

f = @(h) fitAndPredict(X, Y,distr, prspec, prior_inds, prior_grp, h);
%% sample a few samples from the Latin Hypercube design
nInit = 7 * d;
% obsX = lhsdesign(d, nInit)';

obsX = glms.makeHyperParameterGrid(domain, nInit, 'lhs');

obsY = zeros(size(obsX, 1), 1);
for k = 1:size(obsX, 1)
    obsY(k) = f(obsX(k, :));
end

%% initialize the prior
gps = covarianceKernelFactory(1, d);

%% do a litle active learning dance
for k = 1:maxIter
    % ask where to sample next (choose your favorite algorithm)
    %nextX = aoMockus(domain, obsX, obsY, gps);
    nextX = aoKushner(domain, obsX, obsY, gps);

    % evaluate at the suggested point
    nextY = f(nextX);

    % save the measurement pair
    obsX = [obsX; nextX];
    obsY = [obsY; nextY];
end

%% report what has been found
[mv, mloc] = min(obsY);
fprintf('Minimum value: %f found at:\n', mv);
disp(obsX(mloc, :));

S.hyprBin = obsX(mloc, :);
S.obsX = obsX;
S.obsY = obsY;


function nll = fitAndPredict(X,Y,distr, prspec,prior_inds, prior_grp, hyperParameters)

    Cinv = glms.buildPriorCovariance(prspec, prior_inds, prior_grp, hyperParameters);

    [~, ~, S] = glms.getPosteriorWeights(X,Y,Cinv, distr, 'CV', 10, 'bulk', true);
    
    nll = mean(S.testLikelihood);
    
