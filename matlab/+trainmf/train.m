function [h,info] = train(K,y,maxIterations,termFac)
%TRAIN Physical-kernel TRAIn in eta, h=eta.^2, with MATLAB NNLS reference.
% Matrix-free Gauss-Newton/Steihaug trust region. No augmented penalty rows
% and no output peak clipping. The NNLS residual is not a noise estimate.
y=y(:);
[hnnls,floorResidual]=trainmf.nnls(K,y);
roundoff=100*eps*norm(y);
target=max(termFac*floorResidual,roundoff);
info=struct('converged',false,'exit_reason','maximum_iterations', ...
    'nnls_residual',floorResidual,'residual_target',target, ...
    'roundoff_tolerance',roundoff,'iterations',0,'zero_initialization_guard',false);
if ~any(hnnls)
    h=zeros(size(K,2),1);
    info.converged=true;info.exit_reason='zero_nnls_solution';
    info.residual=norm(y);return;
end
seed=min(abs(y));
if seed==0
    % Original min(abs(y)) initialization is identically stationary at zero.
    % Use a homogeneous positive seed only in this exact-zero case.
    seed=norm(y)/sqrt(numel(y));info.zero_initialization_guard=true;
end
h=ones(size(K,2),1)*(1e-10*seed/size(K,2));eta=sqrt(h);
radius0=.01*sqrt(norm(y));radius=radius0;
f=norm(K*h-y)^2;
for it=1:maxIterations
    g=4*eta.*(K'*(K*h-y));
    Hv=@(v)8*eta.*(K'*(K*(eta.*v)));
    step=steihaug(g,Hv,radius);
    candidateEta=abs(eta+step);candidate=candidateEta.^2;
    next=norm(K*candidate-y)^2;
    predicted=g'*step+.5*step'*Hv(step);
    if predicted>=0 || ~isfinite(predicted) || any(~isfinite(step))
        info.exit_reason='stationary_or_numerical_breakdown';break;
    end
    rho=(next-f)/predicted;
    if isfinite(rho) && rho>.01 && next<=f
        eta=candidateEta;h=candidate;f=next;
    end
    if ~isfinite(rho) || rho<=.25
        radius=max(.5*radius,1e-12*radius0);
    elseif rho>=.75
        radius=min(2*radius,radius0);
    end
    info.iterations=it;
    if sqrt(f)<=target,info.exit_reason='residual_target';break;end
end
info.residual=norm(K*h-y);
info.converged=info.residual<=target;
if info.converged,info.exit_reason='residual_target';end
end

function step=steihaug(g,Hv,radius)
step=zeros(size(g));direction=-g;
for it=1:100
    hd=Hv(direction);curvature=direction'*hd;
    dd=direction'*direction;
    if dd==0,return;end
    if curvature<=0,step=boundary(step,direction,radius);return;end
    alpha=-(g'*direction)/curvature;
    candidate=step+alpha*direction;
    if candidate'*candidate>=radius^2
        step=boundary(step,direction,radius);return;
    end
    next=g+alpha*hd;gg=g'*g;
    if gg==0,return;end
    direction=-next+(next'*next/gg)*direction;step=candidate;g=next;
end
end

function step=boundary(step,direction,radius)
sd=step'*direction;dd=direction'*direction;
tau=(sqrt(max(0,sd^2-dd*(step'*step-radius^2)))-sd)/dd;
step=step+tau*direction;
end
