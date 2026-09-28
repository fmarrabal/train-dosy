function [X,D,info] = solve(Y,b,D,options)
%SOLVE Signed TRAIn-MF revision derived from DiffAtOnce signed-v2.
t0=tic;
o=struct('n_components','auto','r_max',4,'max_outer_iter',300,'validation_rows',[], ...
    'max_iter',2000,'term_factor',1.05,'outer_tol',1e-8,'verbose',false);
assert(isstruct(options)&&isscalar(options),'TRAInMF:Options','Scalar options struct required.');
fields=fieldnames(options);
for j=1:numel(fields)
    assert(isfield(o,fields{j}),'TRAInMF:Option','Unsupported option: %s',fields{j});
    o.(fields{j})=options.(fields{j});
end
validateattributes(Y,{'numeric'},{'real','finite','2d','nonempty'});
assert(size(Y,2)<=8192,'TRAInMF:SignalRegions','Select signal regions first (maximum 8192 columns); do not fit a full 65k baseline.');
validateattributes(b,{'numeric'},{'real','finite','vector','nonnegative','numel',size(Y,1)});
validateattributes(D,{'numeric'},{'real','finite','vector','positive'});
Y=double(Y);b=double(b(:));D=double(D(:));
assert(size(Y,1)>=3&&numel(unique(b))>=3,'TRAInMF:Gradients','Need >=3 distinct b values.');
assert(numel(D)>=256&&all(diff(D)>0),'TRAInMF:Grid','Need >=256 increasing positive SI diffusion nodes.');
for name={'r_max','max_outer_iter','max_iter'}
    validateattributes(o.(name{1}),{'numeric'},{'scalar','real','finite','integer','positive'});
end
validateattributes(o.term_factor,{'numeric'},{'scalar','real','finite','>=',1.02,'<=',1.05});
validateattributes(o.outer_tol,{'numeric'},{'scalar','real','finite','positive'});
validateattributes(o.verbose,{'logical','numeric'},{'scalar','binary'});
automatic=(ischar(o.n_components)||isstring(o.n_components))&&strcmp(o.n_components,'auto');
if automatic,ranks=1:min([o.r_max,size(Y,2),numel(D)]);
else
    validateattributes(o.n_components,{'numeric'},{'scalar','real','finite','positive','integer','<=',size(Y,2)});
    ranks=o.n_components;
end
K=exp(-b*D');scale=norm(Y,'fro')/sqrt(numel(Y));
assert(isfinite(scale),'TRAInMF:Scale','Signal scale overflow.');
if scale==0,scale=1;end
y=Y/scale;
nnlsResidual=zeros(1,size(y,2));
for j=1:size(y,2),[~,nnlsResidual(j)]=trainmf.nnls(K,y(:,j));end
floorResidual=norm(nnlsResidual);roundoff=100*eps*norm(y,'fro');
target=max(o.term_factor*floorResidual,roundoff);
trials=cell(1,numel(ranks));best=[];
validation=false(size(b));
if automatic&&numel(ranks)>1
    assert(numel(b)>=8,'TRAInMF:Validation','Automatic comparison of ranks requires >=8 gradients.');
    if isempty(o.validation_rows)
        [~,order]=sort(b);
        indices=unique(round(linspace(2,numel(b)-1,max(2,floor(numel(b)/4)))));
        validation(order(indices))=true;
    else
        assert(islogical(o.validation_rows)&&numel(o.validation_rows)==numel(b), ...
            'TRAInMF:Validation','validation_rows must be a logical vector matching b.');
        validation=o.validation_rows(:);
    end
    assert(nnz(validation)>=2&&nnz(~validation)>=4,'TRAInMF:Validation','Need >=2 validation and >=4 fit rows.');
end
if any(validation)
    yc=y(~validation,:);kc=K(~validation,:);
    floors=zeros(1,size(y,2));
    for j=1:size(y,2),[~,floors(j)]=trainmf.nnls(kc,yc(:,j));end
    candidateTarget=max(o.term_factor*norm(floors),100*eps*norm(yc,'fro'));
else
    yc=y;kc=K;candidateTarget=target;
end
loss=nan(nnz(validation),numel(ranks));usable=false(size(ranks));
for q=1:numel(ranks)
    trial=solveRank(kc,yc,ranks(q),candidateTarget,o);
    trials{q}=rmfield(trial,{'S','A'});
    if isempty(best)||trial.residual<best.residual,best=trial;end
    if any(validation)
        loss(:,q)=mean((K(validation,:)*trial.S*trial.A-y(validation,:)).^2,2);
        usable(q)=trial.seed_failures==0&&trial.subproblem_failures==0&&isempty(trial.inactive_factors);
    end
    if o.verbose,fprintf('TRAIn-MF signed-v2.1: rank %d, fit residual %.6g, target %.6g, %s\n',ranks(q),trial.residual,candidateTarget,trial.status);end
end
selectionResolved=true;
if any(validation)
    meanLoss=mean(loss,1);eligible=meanLoss;eligible(~usable)=Inf;
    if ~any(usable),eligible=meanLoss;selectionResolved=false;end
    [~,bestIndex]=min(eligible);bestLoss=loss(:,bestIndex);
    selectedIndex=bestIndex;
    % Paired one-standard-error rule: prefer fewer predictive shared factors.
    % NNLS per-column residuals are NOT evidence for extra factors.
    for q=1:bestIndex
        difference=loss(:,q)-bestLoss;
        if (usable(q)||~selectionResolved)&&mean(difference)<=std(difference)/sqrt(numel(difference))
            selectedIndex=q;break;
        end
    end
    best=solveRank(K,y,ranks(selectedIndex),target,o);
else
    meanLoss=[];
end
X=best.S*best.A*scale;
info=best;info.A=best.A*scale;
info.protocol='signed-v2.1';info.version='4.0-signed';info.options=o;
info.selected_rank=best.rank;info.n_components=best.rank;
info.rank_trials=trials;info.selection_mode='fixed rank';
if automatic
    info.selection_mode='paired one-standard-error held-out prediction; refit selected rank on all rows';
end
info.validation_rows=find(validation);info.validation_loss_by_gradient=loss;
info.validation_mean_loss=meanLoss;info.rank_selection_resolved=selectionResolved;
if ~selectionResolved,info.success=false;info.status='train_mf_requires_review';end
info.nnls_residual=floorResidual*scale;info.residual_target=target*scale;
info.roundoff_tolerance=roundoff*scale;info.residual=norm(K*X-Y,'fro');
info.relative_residual=info.residual/max(norm(Y,'fro'),realmin);
info.residual_history=best.residual_history*scale;
info.signal_scale=scale;info.time_total=toc(t0);
info.chemical_species_identified=false;
info.criterion='Physical-kernel MATLAB NNLS residual proxy, not independent noise or physical validation.';
info.preprocessing='Signed observations; one restored global RMS scale; no minimum subtraction, clipping, column normalization or penalties.';
info.representation='signal mass per diffusion node';
info.input_negative_count=nnz(Y<0);
info.d_mean=(D'*X)./sum(X,1);[~,imax]=max(X,[],1);info.d_mode=D(imax)';
info.d_mean(sum(X,1)==0)=NaN;info.d_mode(sum(X,1)==0)=NaN;
end

function fit=solveRank(K,y,rank,target,o)
n=size(K,2);p=size(y,2);S=zeros(n,rank);residual=y;selected=false(1,p);
seedFailures=0;subFailures=0;seedColumns=zeros(1,rank);failureDetails={};
for j=1:rank
    energy=sum(residual.^2,1);energy(selected)=-Inf;
    [~,next]=max(energy);selected(next)=true;seedColumns(j)=next;
    [h,d]=trainmf.train(K,y(:,next),o.max_iter,o.term_factor);
    seedFailures=seedFailures+~d.converged;
    if sum(h)>0,S(:,j)=h/sum(h);end
    A=amplitudes(K*S(:,1:j),y);residual=y-K*S(:,1:j)*A;
end
A=amplitudes(K*S,y);err=norm(K*S*A-y,'fro');history=err;
iterations=0;reason='maximum_iterations';
for it=1:o.max_outer_iter
    if err<=target,reason='residual_target';break;end
    before=S*A;
    for j=1:rank
        A=amplitudes(K*S,y);aj=A(j,:);energy=aj*aj';
        if energy==0,continue;end
        R=y-K*S*A+(K*S(:,j))*aj;
        signedTarget=(R*aj')/energy;
        [h,d]=trainmf.train(K,signedTarget,o.max_iter,o.term_factor);
        subFailures=subFailures+~d.converged;
        if ~d.converged&&numel(failureDetails)<10,failureDetails{end+1}=d;end %#ok<AGROW>
        if sum(h)==0,continue;end
        candidate=S;candidate(:,j)=h/sum(h);
        candidateA=amplitudes(K*candidate,y);
        next=norm(K*candidate*candidateA-y,'fro');
        if next<=err,S=candidate;err=next;end
    end
    A=amplitudes(K*S,y);err=norm(K*S*A-y,'fro');history(end+1)=err; %#ok<AGROW>
    iterations=it;
    if err<=target,reason='residual_target';break;end
    if norm(S*A-before,'fro')<=o.outer_tol*max(1,norm(before,'fro'))
        reason='stagnated_before_target';break;
    end
end
success=err<=target&&seedFailures==0&&subFailures==0&&all(sum(A.^2,2)>0);
status='train_mf_requires_review';if success,status='train_mf_discrepancy_reached';end
fit=struct('S',S,'A',A,'rank',rank,'status',status,'success',success, ...
    'residual',err,'residual_history',history,'iterations',iterations, ...
    'exit_reason',reason,'seed_failures',seedFailures,'subproblem_failures',subFailures, ...
    'subproblem_failure_details',{failureDetails}, ...
    'seed_columns',seedColumns,'inactive_factors',find(sum(A.^2,2)==0)');
end

function A=amplitudes(B,y)
A=zeros(size(B,2),size(y,2));
for j=1:size(y,2),A(:,j)=trainmf.nnls(B,y(:,j));end
end
