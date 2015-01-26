%% taking notes on functions that are available from the stats package in matlab

% cross validation
cvfolds = cvpartition(nValidTrials, 'Kfold', nFolds);
cvp = cvpartition(Y, 'k', 10);
fold = 1;
cvfolds.training(1);

% find a DC term
dcTerm = range(models(2).liargs{1})==0;
%%

 
       
       %%
 cvfun = @(Xtrain,Ytrain,Xtest,Ytest) FitAndPredict( ...
        Xtrain,Ytrain,Xtest,Ytest, ...
        priorArges);
    weights = pwts;
    if isempty(weights)
        weights = nan(size(X,1),1);
    end
    if isempty(offset) || isequal(offset,0)
        offset = nan(size(X,1),1);
    end
    if binomialTwoColumn
        response = [nTrials Y];
    else
        response = Y;
    end
    cvDeviance = crossval(cvfun,[weights(:) offset(:) X],response, ...
        'Partition',cvp,'Mcreps',mcreps,'Options',parallelOptions);
    %%


% Head off potential cruft in the command window.
wsIllConditioned2 = warning('off','stats:glmfit:IllConditioned');
wsIterationLimit = warning('off','stats:glmfit:IterationLimit');
wsPerfectSeparation = warning('off','stats:glmfit:PerfectSeparation');
wsBadScaling = warning('off','stats:glmfit:BadScaling');
cleanupIllConditioned2 = onCleanup(@() warning(wsIllConditioned2));
cleanupIterationLimit = onCleanup(@() warning(wsIterationLimit));
cleanupPerfectSeparation = onCleanup(@() warning(wsPerfectSeparation));
cleanupBadScaling = onCleanup(@() warning(wsBadScaling));

%% what if we made prior specs like dspecs


priors = struct('label', [], ...
    'fun', [], ...
    'dfltHyperParams', [], ...
    'nHyperParams', []);

% ridge prior
priors(1).label           = 'Ridge1';
priors(1).fun             = @ridge;
priors(1).dfltHyperParams = .1;
priors(1).nHyperParams    = 1;

% pairwise differences
priors(2).label           = 'Smooth1';
priors(2).fun             = @pairwiseRidge;
priors(2).dfltHyperParams = 10;
priors(2).nHyperParams    = 1;

%% blk diag prior

