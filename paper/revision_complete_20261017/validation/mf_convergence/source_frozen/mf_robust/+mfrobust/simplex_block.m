function [S,meta]=simplex_block(K,Y,A,L,lambda,S0,o)
% Joint S block. Orthogonal QR is an EXACT reduction over ALL frequencies:
% A'=Q*R, ||K*S*A-Y||F^2=||K*S*R'-Y*Q||F^2+constant.
nf=size(Y,2);[nd,r]=size(S0);[Qa,Ra]=qr(A',0);
C=[kron(Ra/sqrt(nf),K);sqrt(lambda)*kron(eye(r),full(L))];
rhs=[reshape(Y*Qa/sqrt(nf),[],1);zeros(size(L,1)*r,1)];
eq=kron(eye(r),ones(1,nd));x0=S0(:);
opts=optimoptions('lsqlin','Algorithm','active-set','Display','off', ...
    'OptimalityTolerance',o.solver_tol,'ConstraintTolerance',o.solver_tol,'MaxIterations',o.solver_max_iter);
[x,~,~,flag,out]=lsqlin(C,rhs,[],[],eq,ones(r,1),zeros(nd*r,1),[],x0,opts);
feasible=@(z)~isempty(z)&&all(isfinite(z))&&min(z)>-1e-8&&max(abs(eq*z-1))<1e-8;
fallback=false;
if flag<=0||~feasible(x)||norm(C*x-rhs)^2>norm(C*x0-rhs)^2+1e-10
    fallback=true;opts.Algorithm='interior-point';
    [xi,~,~,fi,oi]=lsqlin(C,rhs,[],[],eq,ones(r,1),zeros(nd*r,1),[],[],opts);
    if feasible(xi)&&(~feasible(x)||norm(C*xi-rhs)<=norm(C*x-rhs)),x=xi;flag=fi;out=oi;end
end
rejected_infeasible=~feasible(x);
if rejected_infeasible
    % A valid previous iterate is always available. Rejected subproblems are
    % counted as failures; reduced-objective refinement can then continue.
    x=x0;flag=-100;
end
candidate=reshape(max(x,0),nd,r);candidate=candidate./sum(candidate,1);delta=candidate(:)-x0;
v=C*delta;curv=v'*v;der=(C*x0-rhs)'*v;
if curv>0,alpha=min(1,max(0,-der/curv));else,alpha=0;end
S=reshape(x0+alpha*delta,nd,r);
meta=struct('exitflag',flag,'iterations',out.iterations,'fallback',fallback,'step',alpha, ...
    'rejected_infeasible',rejected_infeasible);
end
