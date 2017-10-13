% set up filter
nw = 400;
wts = 2*normpdf(1:nw,nw/2,3)';
fnlin = @nlfuns.exp;
tt = 1:nw;
clf;
plot(tt,wts,'k');
errfun = @(w)(norm(w-wts).^2);  % error function handle

% Make stimuli & simulate response
nstim = 14000;
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

%% Find ML estimate using fminunc compare to glmfit

lfunc = @(w)(glms.neglog.poisson(w,stim,y,fnlin)); % neglogli function handle

opts = optimoptions(@fminunc,'Algorithm','trust-region',...
    'GradObj','on','Hessian','on');

tic;
[wml,nlogli] = fminunc(lfunc,wls,opts);
toc;




plot(tt,wts,'k',tt,[wls,wml]);
legend({'true', 'wls', 'wml'})

%% 
% TODO: something is wrong with AR1 prior
prspec = gpriors.getPriorStruct('smooth');

prior_inds = {1:nw};
prior_grp  = 1;
hyperParameters = [1e2];
Cinv = glms.buildPriorCovariance(prspec, prior_inds, prior_grp, hyperParameters);

pfun=getPosteriorFunctionHandle('poisson');
lfun=@(w) pfun(w, stim, y, Cinv, 1:numel(y));

a=lfun(wmap);
wmap1=fminunc(lfun, wml, opts);
plot(1:nw, [wts wls wml wmap]);
legend({'true','wls', 'ml', 'map1'})   

lambda = 200;
lambdaVect = lambda*[0;ones(numel(wml)-1,1)];
gOptions.maxIter = 2000;
gOptions.verbose = 1; % Set to 0 to turn off output
wmap2=L1General2_PSSas(lfun, wmap1, lambdaVect, gOptions);


plot(1:nw, [wts wml wmap1 wmap2]);
legend({'true','ml', 'map1', 'map2'})   

errfun(wmap1)
errfun(wmap2)
