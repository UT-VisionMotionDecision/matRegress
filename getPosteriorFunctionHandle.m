%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Posterior Function Handle
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fun = getPosteriorFunctionHandle(distr, options)
% distribution
%   'bernoulli'
%   'poisson'
% options (only apply to poisson distribution right now)
%   link 
%       {'canonical', 'log', 'exp'}
%   dtbin

function [fun, options] = getPosteriorFunctionHandle(distr, options)
if nargin < 2
    options.link='exp';
    options.dtbin=1;
end

if ~isfield(options, 'dtbin')
    options.dtbin=1;
end

switch distr
    case 'poisson'
        switch options.link
            case {'canonical', 'log', 'exp'}
                nlfun = @nlfuns.exp;
        end
        fun = @(w,X,Y,Cinv,inds) poissonPosterior(w,X,Y,Cinv, nlfun, inds, options.dtbin);
        
    case 'bernoulli'
        nlfun=@(x) x;
        fun = @(w,X,Y,Cinv,inds) bernoulliPosterior(w,X,Y, Cinv, inds);
end

options.nlfun=nlfun;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Bernoulli Neg-Log Posterior
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f,g,h] = bernoulliPosterior(wts,X,Y,Cinv,inds)

switch nargout
    case 1
        f = glms.neglog.bernoulli(wts,X,Y,inds);
        fp = gpriors.gaussian_zero_mean_inv(wts, Cinv);
        f = f + fp;
    case 2
        [f,g] = glms.neglog.bernoulli(wts,X,Y,inds);
        [fp,gp] = gpriors.gaussian_zero_mean_inv(wts, Cinv);
        f = f + fp;
        g = g + gp;
    case 3
        [f,g, h] = glms.neglog.bernoulli(wts,X,Y,inds);
        [fp,gp, hp] = gpriors.gaussian_zero_mean_inv(wts, Cinv);
        f = f + fp;
        g = g + gp;
        h = h + hp;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Poisson Neg-Log Posterior
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f,g,h] = poissonPosterior(wts,X,Y,Cinv, nlfun, inds, dtbin)

switch nargout
    case 1
        f = glms.neglog.poisson(wts,X,Y,nlfun, inds, dtbin);
        fp = gpriors.gaussian_zero_mean_inv(wts, Cinv);
        f = f + fp;
    case 2
        [f,g] = glms.neglog.poisson(wts,X,Y,nlfun, inds, dtbin);
        [fp,gp] = gpriors.gaussian_zero_mean_inv(wts, Cinv);
        f = f + fp;
        g = g + gp;
    case 3
        [f,g, h] = glms.neglog.poisson(wts,X,Y,nlfun, inds, dtbin);
        [fp,gp, hp] = gpriors.gaussian_zero_mean_inv(wts, Cinv);
        f = f + fp;
        g = g + gp;
        h = h + hp;
end