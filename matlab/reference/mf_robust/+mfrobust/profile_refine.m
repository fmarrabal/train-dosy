function [S,A,meta]=profile_refine(K,Y,S,A,L,o)
% Variable projection of the UNIQUE ridge-NNLS A optimum. The envelope
% gradient is the partial S gradient of the SAME full-frequency objective.
[nd,r]=size(S);nf=size(Y,2);initial=mfrobust.evaluate(K,Y,S,A,L,o);
warm=A;values=[];evaluations=0;eq=kron(eye(r),ones(1,nd));
opts=optimoptions('fmincon','Algorithm','sqp','SpecifyObjectiveGradient',true, ...
    'Display','off','OptimalityTolerance',min(1e-8,o.kkt_tol/10), ...
    'ConstraintTolerance',1e-10,'StepTolerance',1e-12, ...
    'MaxIterations',o.max_outer_iter,'MaxFunctionEvaluations',max(1000,10*o.max_outer_iter),'OutputFcn',@trace);
[z,~,flag,out]=fmincon(@cost,S(:),[],[],eq,ones(r,1),zeros(nd*r,1),[],[],opts);
candidate=reshape(z,nd,r);
assert(min(candidate,[],'all')>=-1e-8&&max(abs(sum(candidate)-1))<1e-8,'FullSpec:ProfileFeasibility','Infeasible profile result');
candidate=max(candidate,0);candidate=candidate./sum(candidate);
aa=mfrobust.nnls_batch(K*candidate,Y,o.lambda_A,warm);value=mfrobust.evaluate(K,Y,candidate,aa,L,o);
accepted=value<=initial+1e-11*max(1,initial);
if accepted,S=candidate;A=aa;end
meta=struct('exitflag',flag,'iterations',out.iterations,'evaluations',evaluations,'values',values, ...
    'first_order_optimality',out.firstorderopt,'accepted',accepted);
    function [f,g]=cost(zs)
        ss=reshape(zs,nd,r);bb=K*ss;warm=mfrobust.nnls_batch(bb,Y,o.lambda_A,warm);
        ee=bb*warm-Y;
        f=sum(ee.^2,'all')/(2*nf)+o.lambda_S*sum((L*ss).^2,'all')/2+o.lambda_A*sum(warm.^2,'all')/(2*nf);
        gg=K'*(ee*warm'/nf)+o.lambda_S*(L'*(L*ss));g=gg(:);evaluations=evaluations+1;
    end
    function stop=trace(~,v,state)
        stop=false;
        if strcmp(state,'iter')
            values(end+1)=v.fval;
            if o.verbose&&mod(v.iteration,25)==0
                fprintf('  profile iter%d F%.8g optimality%.3g, %d MATLAB NNLS batches\n',v.iteration,v.fval,v.firstorderopt,evaluations);
            end
        end
    end
end
