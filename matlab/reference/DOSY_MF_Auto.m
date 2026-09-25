function [X,D,info]=DOSY_MF_Auto(Y,x,D,sigma,innerRows,validationRows,options)
% Automatic shared-factor selection with explicit observational/numerical gates.
% Only innerRows fit candidates; validationRows screen rank and compare loss.
% Final refit uses their union. Any other rows are untouched. Input frequency
% selection MUST be independent of validationRows. Scalar iid Gaussian sigma
% is assumed known; estimated/correlated noise changes the bound's meaning.
% This estimates resolved predictive factors, never a chemical species count.
if nargin<7,options=struct();end
assert(isstruct(options)&&isscalar(options));
here=fileparts(mfilename('fullpath'));addpath(here,fullfile(here,'mf_robust'),fullfile(here,'mf_original','dosy_v4'));
assert(startsWith(which('lsqnonneg'),matlabroot,'IgnoreCase',true),'MATLAB lsqnonneg required.');
n=size(Y,1);x=x(:);D=D(:);innerRows=logical(innerRows(:));validationRows=logical(validationRows(:));
assert(isnumeric(Y)&&isreal(Y)&&ismatrix(Y)&&~isempty(Y)&&all(isfinite(Y),'all'));
assert(isreal(x)&&all(isfinite(x))&&all(x>=0)&&isreal(D)&&all(isfinite(D))&&all(D>0)&&all(diff(D)>0));
assert(numel(x)==n&&numel(innerRows)==n&&numel(validationRows)==n&&~any(innerRows&validationRows));
assert(nnz(innerRows)>=4&&nnz(validationRows)>=3&&numel(D)>=256);
defaults=struct('units','conjugate','lambda_S',1e-8,'lambda_A',1e-8,'n_starts',2, ...
    'seed',9252026,'max_outer_iter',250,'kkt_tol',1e-6,'objective_tol',1e-9, ...
    'solver_tol',1e-10,'solver_max_iter',2000,'verbose',false);
fields=fieldnames(options);
for j=1:numel(fields)
    assert(isfield(defaults,fields{j}),'DOSY_MF_Auto:Option','Unsupported automatic-selector option: %s',fields{j});
    defaults.(fields{j})=options.(fields{j});
end
assert(strcmp(defaults.units,'conjugate'),'DOSY_MF_Auto:Units','Supply conjugate x,D so x*D is dimensionless.');
evidence=DOSY_RankEvidence(Y(validationRows,:),sigma,.01,4);rmax=evidence.candidate_cap;
info=struct('evidence',evidence,'candidate_fits',{{}},'status','unresolved','chemical_species_identified',false,'resolved_only',false);
if rmax==0,X=zeros(numel(D),size(Y,2));info.status='no_rank_resolved';info.selected_rank=0;return;end
loss=nan(nnz(validationRows),rmax);valid=false(1,rmax);records=cell(1,rmax);
for r=1:rmax
    o=defaults;o.n_components=r;
    try
        [xc,~,fit]=DOSY_MF_Robust(Y(innerRows,:),x(innerRows),D,o);
        if fit.kkt.relative>o.kkt_tol
            o.initial_S=fit.S;o.initial_A=fit.A;o.n_starts=1;o.max_outer_iter=750;
            [xr,~,refined]=DOSY_MF_Robust(Y(innerRows,:),x(innerRows),D,o);
            if refined.kkt.relative<=o.kkt_tol||refined.objective<fit.objective,xc=xr;fit=refined;end
        end
        loss(:,r)=mean((exp(-x(validationRows)*D')*xc-Y(validationRows,:)).^2,2);
        valid(r)=fit.kkt.relative<=defaults.kkt_tol;records{r}=fit;
    catch err
        records{r}=struct('status','solver_error','identifier',err.identifier,'message',err.message);
    end
end
selection=DOSY_SelectConverged(loss,valid,1:rmax);info.selection=selection;info.candidate_fits=records;
info.selected_rank=selection.rank;info.status=selection.status;
if ~isfinite(selection.rank),X=[];return;end
o=defaults;o.n_components=selection.rank;rows=innerRows|validationRows;
[X,D,fit]=DOSY_MF_Robust(Y(rows,:),x(rows),D,o);info.fit=fit;
if fit.kkt.relative>o.kkt_tol
    o.initial_S=fit.S;o.initial_A=fit.A;o.n_starts=1;o.max_outer_iter=750;
    [xr,~,refined]=DOSY_MF_Robust(Y(rows,:),x(rows),D,o);
    if refined.kkt.relative<=o.kkt_tol||refined.objective<fit.objective,X=xr;fit=refined;end
    info.fit=fit;
end
if fit.kkt.relative>o.kkt_tol,info.status='unresolved_final_optimization';end
if evidence.search_cap_insufficient,info.status='unresolved_rank_cap';end
info.resolved_only=strcmp(info.status,'selected_converged');
end
