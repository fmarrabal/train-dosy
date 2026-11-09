function d=curvature_design(K,L,D,w,tol)
% Exact algebra for the copied interior second-density-difference operator.
x=log(D(:));w=w(:);v=w.*(x-(w'*x)/sum(w));
d=struct('tolerance',tol,'kernel_ratio',norm(K*v)/max(norm(K,2)*norm(v),realmin), ...
 'operator_residual',norm(L*v)/max(norm(L,'fro')*norm(v),realmin), ...
 'mass_residual',abs(sum(v))/max(norm(v,1),realmin));
d.operator_consistent=d.operator_residual<=tol&&d.mass_residual<=tol;
end
