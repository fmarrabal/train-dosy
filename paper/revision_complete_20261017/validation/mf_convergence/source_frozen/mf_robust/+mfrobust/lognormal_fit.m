function fit=lognormal_fit(Y,x,D,r,seed)
% Variable projection of r unimodal lognormal profiles on >=256 bins.
% Every A update is the supplied MATLAB lsqnonneg. Profiles are shared across
% all supplied columns; for r=2 the r=1 family is explicitly nested.
if nargin<5,seed=[];end
assert(numel(D)>=256&&ismember(r,[1 2]));x=x(:);D=D(:);u=log(D);
du=diff(u);q=[du(1)/2;(du(1:end-1)+du(2:end))/2;du(end)/2];K=exp(-x*D');
scale=max(abs(Y),[],'all');if scale==0,scale=1;end;V=Y/scale;nf=size(V,2);lambda=1e-8;
opts=optimoptions('fmincon','Algorithm','sqp','Display','off','SpecifyObjectiveGradient',true, ...
    'OptimalityTolerance',1e-10,'StepTolerance',1e-12,'MaxIterations',1000,'MaxFunctionEvaluations',10000);
lower=[repmat(u(1),r,1);repmat(log(.06),r,1)];upper=[repmat(u(end),r,1);repmat(log(1.2),r,1)];
if r==1
    initial={[log(2);log(.35)],[log(.5);log(.2)],[log(10);log(.4)]};
else
    assert(~isempty(seed)&&size(seed.S,2)==1);
    mu=seed.parameters(1);w=seed.parameters(2);
    initial={[mu;mu+log(2);w;log(.15)], ...
        [mu-log(1.4);mu+log(1.4);log(.08);log(.08)], ...
        [mu;mu+log(5);w;log(.15)], ...
        [mu-log(3);mu;log(.15);w]};
end
best=Inf;trials=struct([]);
for st=1:numel(initial)
    p0=max(lower,min(upper,initial{st}));
    [p,val,flag,out]=fmincon(@cost,p0,[],[],[],[],lower,upper,[],opts);
    rec=struct('objective',val,'exitflag',flag,'iterations',out.iterations,'optimality',out.firstorderopt);
    if st==1,trials=rec;else,trials(st)=rec;end
    if val<best,best=val;bestp=p;end
end
[~,grad,S,A]=cost(bestp);
pg=grad;pg(bestp<=lower+1e-8)=min(pg(bestp<=lower+1e-8),0);
pg(bestp>=upper-1e-8)=max(pg(bestp>=upper-1e-8),0);
projected_gradient=norm(pg,inf);
assert(projected_gradient<1e-7,'MF256:LocalStationarity','Local profile parameters are not stationary: %.3g',projected_gradient);
[~,order]=sort(bestp(1:r));S=S(:,order);A=A(order,:);
fit=struct('S',S,'A',A*scale,'parameters',[bestp(order);bestp(r+order)],'objective',best, ...
    'prediction',K*S*A*scale,'residual',Y-K*S*A*scale,'trials',trials, ...
    'quadrature',q,'scale',scale,'r',r,'projected_gradient_inf',projected_gradient,'stationarity_tolerance',1e-7);
if r==2
    % Preserve the feasible nested solution if an optimizer misses it.
    value=.5*sum(seed.residual.^2,'all')/(scale^2*nf)+lambda*sum((seed.A/scale).^2,'all')/(2*nf);
    assert(best<=value+1e-8,'MF256:NestedFit','Two-profile solver failed to improve the feasible one-profile solution.');
end
    function [val,g,S,A]=cost(p)
        mu=p(1:r)';sigma=exp(p(r+1:end))';v=(u-mu)./sigma;
        logw=-.5*v.^2;logw=logw-max(logw,[],1);S=q.*exp(logw);S=S./sum(S,1);
        B=K*S;A=mfrobust.nnls_batch(B,V,lambda,[]);E=B*A-V;
        val=(sum(E.^2,'all')+lambda*sum(A.^2,'all'))/(2*nf);
        scoremu=(u-mu)./(sigma.^2);scorewidth=v.^2;
        dmu=S.*(scoremu-sum(S.*scoremu,1));dw=S.*(scorewidth-sum(S.*scorewidth,1));
        partial=K'*(E*A')/nf;g=[sum(partial.*dmu,1)';sum(partial.*dw,1)'];
    end
end
