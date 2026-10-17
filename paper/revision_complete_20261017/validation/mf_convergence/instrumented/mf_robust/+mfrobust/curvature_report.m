function d=curvature_report(A,design)
% A floating-point sufficient-condition diagnostic, not physical uniqueness.
tol=design.tolerance;scale=max(A,[],'all');r=size(A,1);
if scale==0,s=zeros(min(size(A)),1);else,s=svd(A/scale);end
ratio=0;nr=0;
if ~isempty(s)&&s(1)>0
 nr=nnz(s>tol*s(1));if numel(s)==r,ratio=s(end)/s(1);end
end
d=design;d.requested_rank=r;d.numeric_row_rank=nr;d.amplitude_ratio=ratio;
d.supported=design.operator_consistent&&design.kernel_ratio>tol&&nr==r;
d.stage='';d.iteration=0;d.fallback_used=false;
end
