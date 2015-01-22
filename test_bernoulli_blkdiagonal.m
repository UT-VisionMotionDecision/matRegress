
% 1.  Set up simulated example

% set up filter
nw = 50; % number of coeffs in filter
wts = 3*normpdf(1:nw,nw/2,sqrt(nw)/2)';  % linear filter
b = -1; % constant (DC term)

% Make stimuli & simulate response
nstim = 2000;
stim = 1*(randn(nstim,nw));
xproj = stim*wts+b;
pp = logsig(xproj);
yy = rand(nstim,1)<pp;

% -- make plot ---
tt = 1:nw;
figure(1); clf
subplot(212);
plot(tt,wts,'k');
title('true filter');
subplot(211);
xpl = min(xproj):.1:max(xproj);
plot(xproj,yy,'.',xpl,logistic(xpl), 'k');
xlabel('input'); ylabel('response');
fprintf('mean rate = %.1f (%d ones)\n', sum(yy)/nstim, sum(yy));

errfun = @(w)(norm(w-wts).^2);  % error function handle

%% 2. Compute (standard) linear regression estimates
xx = [stim, ones(nstim,1)];  % regressors

% LS estimate
wls = xx\yy;
% MAP estimate
lam = 10000; % ridge parameter
wmap0 = (xx'*xx + lam*speye(nw+1))\(xx'*yy);

subplot(212);
plot(tt,wts,'k',tt,wls(1:nw)/norm(wls(1:nw))*norm(wts),...
    tt,wmap0(1:nw)/norm(wmap0(1:nw))*norm(wts));
legend('original', 'LS', 'ridge');

%%

smoothness = 5000;
% shrinkage  = 100;

mstruct.neglogli  = @neglogli.bernoulli; % neg log-likelihood function
mstruct.neglogpr  = @neglogprior.gaussian_zero_mean_inv;
mstruct.liargs    = {xx,yy}; % args for likelihood function

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


subplot(212);
plot(tt,wts,'k', tt, wmap(1:nw));

axis tight;
legend('true','MAP');

