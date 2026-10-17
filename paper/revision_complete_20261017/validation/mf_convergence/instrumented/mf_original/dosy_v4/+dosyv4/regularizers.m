function [L,q,Lf,edges] = regularizers(D,nf,o)
% S is probability MASS. rho=S./q is density in u=ln(D).
% Interior curvature/slope quadrature; no imposed reflective boundary prior.
u=log(D(:)); n=numel(u); du=diff(u);
q=[du(1)/2;(du(1:end-1)+du(2:end))/2;du(end)/2];
if o.regularizer_order==1
    L=spdiags(1./sqrt(du),0,n-1,n-1)*diff(speye(n),1,1)*spdiags(1./q,0,n,n);
elseif n>=3
    avg=(du(1:end-1)+du(2:end))/2;
    H=sparse(n-2,n);
    for j=2:n-1
        H(j-1,j-1:j+1)=[1/du(j-1),-1/du(j-1)-1/du(j),1/du(j)]/avg(j-1);
    end
    L=spdiags(sqrt(avg),0,n-2,n-2)*H*spdiags(1./q,0,n,n);
else
    assert(o.lambda_S==0,'DOSY:Grid','Curvature regularization needs at least 3 nodes.');
    L=sparse(0,n);
end
Lf=sparse(0,nf); edges=zeros(0,2);
if o.lambda_frequency==0, return; end
f=o.frequency_axis(:);
assert(isnumeric(f)&&isreal(f)&&numel(f)==nf&&all(isfinite(f))&&numel(unique(f))==nf, ...
    'DOSY:FrequencyAxis','Spectral regularization requires distinct finite frequency coordinates.');
gap=o.max_spectral_gap;
assert(isnumeric(gap)&&isreal(gap)&&isscalar(gap)&&isfinite(gap)&&gap>0, ...
    'DOSY:SpectralGap','Specify a finite positive max_spectral_gap in frequency-axis units.');
regions=o.spectral_regions;
if isempty(regions), regions=ones(nf,1); end
assert(isnumeric(regions)&&numel(regions)==nf&&all(isfinite(regions)),'DOSY:Regions','Invalid spectral_regions.');
regions=regions(:); [fs,ix]=sort(f); gaps=diff(fs);
take=find(gaps<=gap & regions(ix(1:end-1))==regions(ix(2:end)));
edges=[ix(take),ix(take+1)]; m=numel(take);
Lf=sparse(repelem((1:m)',2),reshape(edges',[],1), ...
    reshape([-1./sqrt(gaps(take)),1./sqrt(gaps(take))]',[],1),m,nf);
end
