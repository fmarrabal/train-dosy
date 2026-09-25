function [A,meta]=nnls_batch(B,Y,lambda,~)
% MATLAB Lawson-Hanson NNLS on EVERY frequency column, original signed Y.
% ||B*a-y||^2 + lambda||a||^2 = ||[B;sqrt(lambda)I]a-[y;0]||^2.
% The user-supplied lsqnonneg source is identical to the installed function.
[m,r]=size(B);nf=size(Y,2);
assert(size(Y,1)==m&&all(isfinite(B),'all')&&all(isfinite(Y),'all'));
assert(isscalar(lambda)&&isfinite(lambda)&&lambda>=0);
C=[B;sqrt(lambda)*eye(r)];rhs=[Y;zeros(r,nf)];
op=optimset('Display','off');A=zeros(r,nf);failures=0;iterations=0;
for j=1:nf
    [a,~,~,flag,out]=lsqnonneg(C,rhs(:,j),op);
    if flag~=1||any(~isfinite(a))||any(a<0),failures=failures+1;end
    A(:,j)=a;iterations=iterations+out.iterations;
end
assert(failures==0,'MF256:NNLSFailure','MATLAB NNLS failed on %d columns.',failures);
G=B'*Y;grad=B'*(B*A-Y)+lambda*A;scale=max(1,max(abs(G),[],1));
dual=max(max(-grad,0),[],1);comp=max(abs(A.*grad),[],1)./max(1,max(A,[],1));
meta=struct('kkt',max(max(dual,comp)./scale),'fallback_columns',0, ...
    'solver','MATLAB lsqnonneg','calls',nf,'failures',failures,'iterations',iterations);
end
