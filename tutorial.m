% test basic logistic regression code on 1D simulated example

% set up filter
nw = 100;
wts = 2*normpdf(1:nw,nw/2,3)';
fnlin = @nlfuns.exp;
tt = 1:nw;
clf;
plot(tt,wts,'k');
errfun = @(w)(norm(w-wts).^2);  % error function handle

% Make stimuli & simulate response
nstim = 400;
stim = 1*(randn(nstim,nw));
xproj = stim*wts;
pp = fnlin(xproj);
y = poissrnd(pp);
fprintf('mean rate = %.1f (%d spikes)\n', sum(y)/nstim, sum(y));

% Compute linear regression solution
wls = stim\y;
wls = wls/norm(wls)*norm(wts); % normalize so vector norm is correct
figure(1); clf
plot(tt,wts,'k',tt,wls);


%% Find ML estimate using fminunc
lfunc = @(w)(neglogli.poisson(w,stim,y,fnlin)); % neglogli function handle

opts = optimoptions(@fminunc,'Algorithm','trust-region',...
    'GradObj','on','Hessian','on');

tic;
[wml,nlogli] = fminunc(lfunc,wls,opts);
toc;

tic;
b = glmfit(stim, y, 'poisson');
toc

[B FitInfo] = lassoglm(stim,y,'poisson','CV',10,'Alpha', .2);

plot(tt,wts,'k',tt,[wls,wml, b(2:end), B(:,FitInfo.IndexMinDeviance)]);
legend({'true', 'wls', 'wml', 'b'})

%%

% Examine the cross-validation plot to see the effect of the Lambda regularization parameter.

lassoPlot(B,FitInfo,'plottype','CV');



%% Find MAP estimate with a fixed prior hyper parameter
smoothness = 1000;
% shrinkage  = 100;

mstruct.neglogli  = @neglogli.poisson; % neg log-likelihood function
mstruct.neglogpr  = @neglogprior.gaussian_zero_mean_inv;
mstruct.liargs    = {stim,y,fnlin}; % args for likelihood function

% % prior arguments
% mstruct.priors    = {@gpriors.smooth, @gpriors.ridge};
% mstruct.hyprprs   = {smoothness, shrinkage};
% mstruct.indices   = {1:15, 15:30};

% prior arguments
mstruct.priors    = {@gpriors.smooth};
mstruct.hyprprs   = {smoothness};
mstruct.indices   = {1:nw};

Cinv = gpriors.blkdiagPrior(wls, mstruct.priors, mstruct.hyprprs, mstruct.indices);

mstruct.priargs   = {Cinv}; % additional prior arguments


lfpost = @(w)(neglogprior.posterior(w,mstruct)); % posterior
HessCheck(lfpost,wls);  % check gradient & Hessian

tic;
[wmap,nlogpost,H] = fminunc(lfpost,wls*.1,opts);
toc

plot(tt,wts,'k',tt,[wls,wml,wmap]);
axis tight;
legend('true','LS','ML','MAP');
% ebr = 3*sqrt(diag(inv(H)));


%% Search space of hyperparameters
nFolds = 5;
isRandomized = 1;
folds = tools.xvalidationIdx(nstim, nFolds, isRandomized);
% gridparams = {[1 10 100 1000 2000 5000 10000], [1 10 100 1000]};
gridparams = {[1 10 100 1000 2000 5000 10000]};

[wmaps model] = tools.cvglm(mstruct, folds, gridparams);

%%  Examine results

Errs = [errfun(wml), errfun(mean(model.foldMaxWts,2)) errfun(B(:,FitInfo.IndexMinDeviance))]



% Evidences = [max(evids(:)), logevid,logevid2]


plot(tt,wts,'k',tt,[wls,wml,mean(model.foldMaxWts,2)]);
axis tight;
legend('true','LS','ML','CV');

