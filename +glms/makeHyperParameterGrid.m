function hgrid = makeHyperParameterGrid(domain, ngridpoints, gridType)
% hgrid = makeHyperParameterGrid(domain, nGridPoints, gridType)

nhyp = size(domain,1);
switch gridType
    case 'lhs'
        hgrid = lhsdesign(ngridpoints,nhyp);
    case 'uniform'
        error('implement me')
end


for khp = 1:nhyp
    hgrid(:,khp) = hgrid(:,khp)*diff(domain(khp,:)) + domain(khp,1);
end