function evidence=DOSY_RankEvidence(Y,sigma,alpha,maxRank)
% Conservative count of noise-resolved matrix directions, not chemical species.
% Y must use rows independent of any data-driven frequency mask. IID Gaussian
% noise with known common sigma is required for the stated norm bound.
if nargin<3,alpha=.01;end
if nargin<4,maxRank=4;end
assert(isreal(Y)&&ismatrix(Y)&&all(isfinite(Y),'all')&&~isempty(Y));
assert(isscalar(sigma)&&isfinite(sigma)&&sigma>0);
assert(isscalar(alpha)&&alpha>0&&alpha<1);
[n,p]=size(Y);s=svd(double(Y),'econ');
bound=sigma*(sqrt(n)+sqrt(p)+sqrt(2*log(1/alpha)));
resolved=sum(s>bound);
evidence=struct('singular_values',s,'noise_operator_bound',bound, ...
    'resolved_rank',resolved,'candidate_cap',min(maxRank,resolved), ...
    'search_cap_insufficient',resolved>maxRank,'alpha',alpha, ...
    'sigma',sigma,'noise_model','known iid Gaussian', ...
    'ratios',s/bound,'rank_at_sigma_minus20pct',sum(s>.8*bound), ...
    'rank_at_sigma_plus20pct',sum(s>1.2*bound), ...
    'interpretation','Evidence lower count for signal matrix rank; undetected directions may exist. Not an upper bound on species.');
end
