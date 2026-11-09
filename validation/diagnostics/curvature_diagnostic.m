function [report,L,v]=curvature_diagnostic(b,D,A,lambdaS,tolerance)
%CURVATURE_DIAGNOSTIC Post-hoc check for the submitted interior-curvature MF.
% Does NOT instrument or change the frozen solver. The exact conditional
% injectivity condition reduces to Kv~=0 and full ROW rank of A. Numeric
% support at a declared tolerance is not exact rank or physical uniqueness.
if nargin<5,tolerance=1e-10;end
validateattributes(b,{'numeric'},{'real','finite','vector','nonnegative','nonempty'});
validateattributes(D,{'numeric'},{'real','finite','vector','positive','nonempty'});
validateattributes(A,{'numeric'},{'real','finite','2d','nonnegative','nonempty'});
validateattributes(lambdaS,{'numeric'},{'scalar','real','finite','positive'});
validateattributes(tolerance,{'numeric'},{'scalar','real','finite','>',0,'<',1});
b=double(b(:));D=double(D(:));A=double(A);x=log(D);q=numel(D);
assert(q>=3&&all(diff(D)>0),'CurvatureAudit:Grid','Need >=3 increasing positive nodes.');
dx=diff(x);w=[dx(1)/2;(dx(1:end-1)+dx(2:end))/2;dx(end)/2];
width=(dx(1:end-1)+dx(2:end))/2;
H=sparse(q-2,q);
for h=2:q-1
    H(h-1,h-1:h+1)=[1/dx(h-1),-1/dx(h-1)-1/dx(h),1/dx(h)]/width(h-1);
end
L=spdiags(sqrt(width),0,q-2,q-2)*H*spdiags(1./w,0,q,q);
center=(w'*x)/sum(w);v=w.*(x-center);K=exp(-b*D');
largest=max(A,[],'all');
if largest==0
    singular=zeros(min(size(A)),1);
else
    singular=svd(A/largest);
end
r=size(A,1);rowRatio=0;numericRank=0;
if ~isempty(singular)&&singular(1)>0
    numericRank=nnz(singular>tolerance*singular(1));
    if numel(singular)==r,rowRatio=singular(end)/singular(1);end
end
kvRatio=norm(K*v)/(norm(K,2)*norm(v));
nullResidual=norm(L*v)/(norm(L,'fro')*norm(v));
massResidual=abs(sum(v))/max(norm(v,1),realmin);
operatorConsistent=nullResidual<=tolerance&&massResidual<=tolerance;
supported=operatorConsistent&&numericRank==r&&kvRatio>tolerance;
status='conditional_injectivity_unresolved_at_tolerance';
if supported,status='conditional_injectivity_numerically_supported';end
report=struct('scope','Post-hoc algebraic diagnostic; frozen solver unchanged', ...
    'condition','Kv nonzero and A full row rank for the specified interior curvature', ...
    'lambda_S',lambdaS,'tolerance',tolerance,'rank_requested',r, ...
    'amplitude_numeric_row_rank',numericRank,'amplitude_singular_value_ratio',rowRatio, ...
    'kernel_null_direction_ratio',kvRatio,'curvature_null_direction_residual',nullResidual, ...
    'mass_null_direction_residual',massResidual,'operator_consistent',operatorConsistent, ...
    'conditional_injectivity_numerically_supported',supported,'status',status, ...
    'outer_factorization_uniqueness_claimed',false,'physical_identification_claimed',false);
end
