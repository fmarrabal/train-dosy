function [X,D,info]=DOSY_MF_Robust(Y,x,D,options)
% Full-frequency constrained factorization, research reference 5.1.
% .5/nf||K*S*A-Y/q||F^2 + lambdaS/2||L*S||F^2 + lambdaA/(2nf)||A||F^2
% S>=0, sum(S,1)=1, A>=0. No spectral selection or binning. Real signed Y.
if nargin<4,options=struct();end
here=fileparts(mfilename('fullpath'));addpath(fullfile(here,'..','mf_original','dosy_v4'));
assert(numel(D)>=256,'MF256:Grid','The MF reconstruction requires at least 256 diffusion bins.');
assert(~isfield(options,'sigma')||isempty(options.sigma),'FullSpec:Weights','This implementation uses homogeneous relative weighting.');
assert(~isfield(options,'lambda_frequency')||options.lambda_frequency==0,'FullSpec:SpectralPenalty','No spectral smoothness penalty in this reference.');
p=dosyv4.prepare(Y,x,D,options);o=p.options;K=p.K;V=p.Y;L=p.L;D=p.D;
t=tic;[nd,r,nf]=deal(numel(D),o.n_components,size(Y,2));starts=struct([]);best=[];bestval=Inf;
for start=1:o.n_starts
    stream=RandStream('mt19937ar','Seed',mod(o.seed+104729*(start-1),2^32));
    S=zeros(nd,r);A=zeros(r,nf);
    if start==1 && ~isempty(o.initial_S)
        S=o.initial_S;
    elseif start==1
        % Greedy residual energy across ALL frequency columns, initialization only.
        residual=V;used=[];
        for k=1:r
            gains=sum(max(K'*residual,0).^2,2)./max(sum(K.^2,1)',realmin);gains(used)=-Inf;
            [~,ix]=max(gains);used(end+1)=ix;S(ix,k)=1; %#ok<AGROW>
            aa=mfrobust.nnls_batch(K*S(:,1:k),V,o.lambda_A,[]);residual=V-K*S(:,1:k)*aa;
        end
    else
        u=log(D);span=u(end)-u(1);
        for k=1:r
            center=u(1)+span*((k-1)+rand(stream))/r;width=span*(.025+.10*rand(stream));
            s=p.quadrature.*exp(-.5*((u-center)/width).^2);S(:,k)=s/sum(s);
        end
    end
    if start==1&&~isempty(o.initial_A),A=o.initial_A/p.scale;end
    [val,parts]=mfrobust.evaluate(K,V,S,A,L,o);history=val;termhist=parts;kh=[];status='max_iter';fallbacks=0;subfail=0;
    for iteration=1:min(8,o.max_outer_iter)
        before=val;
        [An,am]=mfrobust.nnls_batch(K*S,V,o.lambda_A,A);fallbacks=fallbacks+am.fallback_columns;
        [av,ap]=mfrobust.evaluate(K,V,S,An,L,o);
        if av>val+1e-10*max(1,val),status='A_objective_failure';break;end
        A=An;val=av;history(end+1)=val;termhist(end+1,:)=ap; %#ok<AGROW>
        [Sn,sm]=mfrobust.simplex_block(K,V,A,L,o.lambda_S,S,o);subfail=subfail+(sm.exitflag<=0);
        [sv,sp,kkt]=mfrobust.evaluate(K,V,Sn,A,L,o);
        if sv>val+1e-10*max(1,val),status='S_objective_failure';break;end
        S=Sn;val=sv;history(end+1)=val;termhist(end+1,:)=sp;kh(end+1)=kkt.relative; %#ok<AGROW>
        if abs(before-val)/max(1,abs(before))<=o.objective_tol && kkt.relative<=o.kkt_tol,status='converged';break;end
    end
    [val,~,kkt]=mfrobust.evaluate(K,V,S,A,L,o);
    refinement=struct();
    if kkt.relative>o.kkt_tol
        [S,A,refinement]=mfrobust.profile_refine(K,V,S,A,L,o);
        [val,parts,kkt]=mfrobust.evaluate(K,V,S,A,L,o);
        history(end+1)=val;termhist(end+1,:)=parts; %#ok<AGROW>
        if kkt.relative<=o.kkt_tol,status='converged';else,status='profile_not_stationary';end
    end
    record=struct('status',status,'objective',val,'kkt',kkt,'iterations',iteration,'history',history, ...
        'term_history',termhist,'kkt_history',kh,'S',S,'A',A*p.scale,'fallback_columns',fallbacks,'S_solver_nonpositive_flags',subfail,'profile_refinement',refinement);
    if start==1,starts=record;else,starts(start)=record;end
    good=kkt.relative<=o.kkt_tol; oldgood=~isempty(best)&&best.kkt.relative<=o.kkt_tol;
    if isempty(best)||(good&&~oldgood)||(good==oldgood&&val<bestval),bestval=val;best=record;end
    if o.verbose,fprintf('MF MATLAB-NNLS start %d: F %.8g KKT %.3g %s, %d iters, %.1fs\n',start,val,kkt.relative,status,iteration,toc(t));end
end
[~,order]=sort(log(D)'*best.S);S=best.S(:,order);A=best.A(order,:);X=S*A;
info=struct('version','5.1-feasible-stationary-selection','status',best.status,'S',S,'A',A,'kkt',best.kkt, ...
    'objective',best.objective,'history',best.history,'term_history',best.term_history,'iterations',best.iterations, ...
    'starts',starts,'signal_scale',p.scale,'options',options,'quadrature',p.quadrature,'x',p.b,'D',D, ...
    'n_frequencies',nf,'all_frequencies_used',true,'elapsed_seconds',toc(t));
B=K*S;Bn=B./max(vecnorm(B),realmin);An=A./max(vecnorm(A,2,2),realmin);
info.decay_cosine=Bn'*Bn;info.spectral_cosine=An*An';info.geometric_kappa=exp(log(D)'*S);
info.identifiability='Not chemically certified; r is a shared predictive factor count, S may be multimodal.';
info.frequency_component_fraction=A./max(sum(A,1),realmin);
info.full_prediction=K*X+p.baseline;info.full_residual=info.full_prediction-p.raw_Y;
end
