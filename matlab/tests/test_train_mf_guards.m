function report = test_train_mf_guards(outputFile)
%TEST_TRAIN_MF_GUARDS Adversarial regression for signed-v2.2 diagnostics.
% Synthetic iid Gaussian sigma is known independently of the fit. Truth is
% used only for evaluation, never passed to the selector. Passing this suite
% is not experimental validation or a proof of physical identifiability.
% Run with MATLAB + Optimization Toolbox, e.g.:
%   addpath('matlab/tests'); test_train_mf_guards('guards.json');
here=fileparts(mfilename('fullpath'));addpath(fileparts(here));
previousRng=rng;restoreRng=onCleanup(@()rng(previousRng)); %#ok<NASGU>
rng(28092026,'twister');
D=logspace(log10(.02e-9),log10(5e-9),256)';
b=12e9*linspace(0,1,28)'.^2;K=exp(-b*D');
bt=12e9*linspace(.025,.975,13)'.^2;Kt=exp(-bt*D');
checks=struct('name',{},'passed',{});shared=cell(1,3);
o=struct('r_max',4,'max_outer_iter',400,'sigma',.001);
for trueRank=1:3
    centers=[.22,.75,2.5]*1e-9;S=zeros(numel(D),trueRank);
    for j=1:trueRank
        S(:,j)=exp(-.5*(log(D/centers(j))/.18).^2);
        S(:,j)=S(:,j)/sum(S(:,j));
    end
    A=zeros(trueRank,6*trueRank);
    for j=1:trueRank,A(j,(j-1)*6+(1:6))=[1,.7,1.3,.8,1.1,.9];end
    if trueRank>1,A(:,end-2:end)=A(:,end-2:end)+.25;end
    truth=S*A;clean=K*truth;Y=clean+o.sigma*randn(size(clean));original=Y;
    [X,~,fit]=TRAIn_DOSY_MF_Signed(Y,b,D,o);
    diagnosticContract(fit,sprintf('shared_%d',trueRank));
    meanTruth=D'*truth./sum(truth,1);meanFit=D'*X./sum(X,1);
    meanError=max(abs(meanFit./meanTruth-1));
    predictionError=norm(Kt*(X-truth),'fro')/norm(Kt*truth,'fro');
    check(isequal(Y,original),sprintf('shared_%d_input_preserved',trueRank));
    check(all(isfinite(X),'all')&&all(X>=0,'all'),sprintf('shared_%d_feasible',trueRank));
    check(all(diff(fit.residual_history)<=1e-11),sprintf('shared_%d_residual_monotone',trueRank));
    check(fit.selected_rank==trueRank&&fit.active_rank==trueRank&&fit.rank_selection_resolved, ...
        sprintf('shared_%d_predictive_rank',trueRank));
    check(meanError<.08&&predictionError<.015,sprintf('shared_%d_truth_accuracy',trueRank));
    check(fit.model_compatible&&~any(fit.boundary_hit(:)),sprintf('shared_%d_noise_and_grid_compatible',trueRank));
    % TRAIn may legitimately fail its stricter numerical stopping condition:
    % report that outcome; do not convert accuracy into claimed convergence.
    shared{trueRank}=struct('true_rank',trueRank,'max_mean_D_relative_error',meanError, ...
        'clean_held_out_relative_error',predictionError,'diagnostics',compact(fit));
    if trueRank==1,firstY=Y;firstX=X;firstFit=fit;end
end

% Changing intensity units must scale the independent sigma too.
scaledOptions=o;scaledOptions.sigma=7*o.sigma;
[scaled,~,sf]=TRAIn_DOSY_MF_Signed(7*firstY,b,D,scaledOptions);
diagnosticContract(sf,'scaled');
scalePrediction=norm(K*(scaled/7-firstX),'fro')/norm(firstY,'fro');
check(sf.selected_rank==firstFit.selected_rank&&sf.model_compatible==firstFit.model_compatible ...
    &&isequal(sf.boundary_hit,firstFit.boundary_hit)&&scalePrediction<1e-4,'known_sigma_scale_invariance');

% Removing independently supplied sigma must not claim model compatibility.
unknownOptions=rmfield(o,'sigma');unknownOptions.n_components=1;
[~,~,unknown]=TRAIn_DOSY_MF_Signed(firstY,b,D,unknownOptions);
diagnosticContract(unknown,'unknown_sigma');
check(~unknown.success&&isnan(unknown.model_compatible),'unknown_sigma_never_validates_model');

% Directly reproduce the misleading success cases from the 2026-10-02 audit.
rng(20261002,'twister');b24=12e9*linspace(0,1,24)'.^2;K24=exp(-b24*D');
boundary=cell(1,2);sigma=.0003;
for j=1:2
    truthD=[.005,8]*1e-9;actualD=truthD(j);
    Y=exp(-b24*actualD)*[1,.7,1.2,.5]+sigma*randn(24,4);
    [X,~,f]=TRAIn_DOSY_MF_Signed(Y,b24,D,struct('r_max',3,'sigma',sigma));
    diagnosticContract(f,sprintf('outside_grid_%d',j));
    check(~f.success&&~f.model_compatible,sprintf('outside_grid_%d_rejected_against_noise',j));
    check(any(f.boundary_hit(:)),sprintf('outside_grid_%d_boundary_reported',j));
    noiseUnits=norm(K24*X-Y,'fro')/(sigma*sqrt(numel(Y)));
    boundary{j}=struct('true_D',actualD,'mean_D',f.d_mean,'rmse_in_noise_units',noiseUnits, ...
        'diagnostics',compact(f));
end

% The null model must compete with r>=1, including single-column input.
rng(2102026,'twister');noiseSigma=.01;
[noiseX,~,noise]=TRAIn_DOSY_MF_Signed(noiseSigma*randn(24,8),b24,D, ...
    struct('r_max',3,'sigma',noiseSigma));
diagnosticContract(noise,'pure_noise');
check(noise.selected_rank==0&&noise.active_rank==0&&all(noiseX==0,'all'),'pure_noise_selects_null');
check(all(isnan(noise.d_mean))&&all(isnan(noise.d_mode)),'pure_noise_has_no_diffusion_estimate');
nulls=cell(1,2);
for j=1:2
    columnCounts=[1,3];p=columnCounts(j);
    [zeroX,~,z]=TRAIn_DOSY_MF_Signed(zeros(24,p),b24,D,struct('r_max',3,'sigma',noiseSigma));
    diagnosticContract(z,sprintf('zero_%d_columns',p));
    check(z.selected_rank==0&&z.active_rank==0&&all(zeroX==0,'all'),sprintf('zero_%d_columns_null',p));
    check(all(isnan(z.d_mean))&&all(isnan(z.d_mode)),sprintf('zero_%d_columns_no_diffusion',p));
    nulls{j}=compact(z);
end

% A clearly incompatible signed baseline must not be approved as no signal.
[~,~,negative]=TRAIn_DOSY_MF_Signed(-ones(24,3),b24,D,struct('r_max',3,'sigma',noiseSigma));
diagnosticContract(negative,'negative_baseline');
check(~negative.success&&~negative.model_compatible,'negative_baseline_rejected');

% Fixed-rank intent cannot silently start automatic hold-out rank selection.
autoFalseError='';
try
    TRAIn_DOSY_MF(firstY,b,[.02 5 256],struct('auto_r',false,'r_max',3,'sigma',o.sigma));
catch e
    autoFalseError=e.identifier;
end
check(strcmp(autoFalseError,'TRAInMF:FixedRank'),'auto_false_requires_explicit_rank');
[~,~,fixed]=TRAIn_DOSY_MF(firstY,b,[.02 5 256], ...
    struct('auto_r',false,'n_components',1,'sigma',o.sigma));
diagnosticContract(fixed,'fixed_rank');
check(fixed.selected_rank==1&&isempty(fixed.validation_rows),'fixed_rank_does_not_select_automatically');

% Exact noiseless signal can stagnate in a nonconvex algorithm. The contract
% must remain truthful regardless of whether this particular case converges.
[~,~,exact]=TRAIn_DOSY_MF_Signed(exp(-b24*.7e-9)*[1,.7,1.2,.5],b24,D,struct('r_max',3));
diagnosticContract(exact,'noiseless');
check(~exact.success,'noiseless_without_sigma_is_not_physical_validation');
check(~exact.rank_selection_resolved||exact.active_rank==exact.selected_rank,'inactive_rank_is_not_resolved');

% The old min(abs(y)) seed became zero after multiplication by 1e-10/n.
tail=exp(-b24*.7e-9);tail(end)=1e-320;
[h,tf]=trainmf.train(K24,tail,2000,1.05);
tailResidual=norm(K24*h-tail)/norm(tail);
check(all(isfinite(h))&&all(h>=0)&&sum(h)>0,'subnormal_tail_positive_finite_solution');
check(tf.zero_initialization_guard&&(tf.iterations>0||tf.converged),'subnormal_tail_initialization_guard');
check(tailResidual<.01,'subnormal_tail_forward_fit');

% Reject invalid noise scales; zero does not establish exact noise knowledge.
badSigma={0,-1,Inf,NaN,[.001 .002]};invalidRejected=false(size(badSigma));
for j=1:numel(badSigma)
    try
        TRAIn_DOSY_MF_Signed(firstY,b,D,struct('n_components',1,'sigma',badSigma{j}));
    catch
        invalidRejected(j)=true;
    end
end
check(all(invalidRejected),'invalid_sigma_rejected');

report=struct('protocol','signed-v2.2','matlab_version',version, ...
    'checks',checks,'checks_passed',all([checks.passed]),'check_count',numel(checks), ...
    'shared_profiles',{shared},'boundary',{boundary},'null_data',{nulls}, ...
    'pure_noise',compact(noise),'negative_baseline',compact(negative), ...
    'unknown_sigma',compact(unknown),'fixed_rank',compact(fixed),'noiseless',compact(exact), ...
    'amplitude_scale_prediction_difference',scalePrediction,'auto_false_error',autoFalseError, ...
    'subnormal_tail',struct('relative_residual',tailResidual,'fit',tf), ...
    'scope','Synthetic numerical and diagnostic regression; no experimental or chemical validation');
if nargin>0
    f=fopen(outputFile,'w');assert(f>=0);closeFile=onCleanup(@()fclose(f)); %#ok<NASGU>
    fprintf(f,'%s',jsonencode(report,PrettyPrint=true));
end
failed={checks(~[checks.passed]).name};
assert(isempty(failed),'TRAInMFTest:Guards','Failed adversarial checks: %s',strjoin(failed,', '));
fprintf('PASS: %d signed-v2.2 adversarial checks; numerical convergence recorded separately.\n',numel(checks));

    function check(condition,name)
        checks(end+1)=struct('name',name,'passed',logical(condition)); %#ok<AGROW>
        if ~condition,fprintf(2,'FAIL: %s\n',name);end
    end

    function diagnosticContract(f,label)
        required={'numerical_converged','model_compatible','rank_selection_resolved', ...
            'active_rank','selected_rank','boundary_hit'};
        assert(all(isfield(f,required)),'TRAInMFTest:Contract','Missing signed-v2.2 diagnostics for %s.',label);
        check(strcmp(f.protocol,'signed-v2.2'),[label '_protocol']);
        check(~f.success||(f.numerical_converged&&f.model_compatible&&f.rank_selection_resolved ...
            &&~any(f.boundary_hit(:))),[label '_success_requires_all_guards']);
        check(f.active_rank>=0&&f.active_rank<=f.selected_rank ...
            &&f.active_rank==fix(f.active_rank),[label '_rank_consistency']);
        check(~f.rank_selection_resolved||f.active_rank==f.selected_rank,[label '_resolved_rank_is_active']);
        check(~f.chemical_species_identified,[label '_no_chemical_identification_claim']);
    end
end

function s=compact(f)
names={'protocol','status','success','numerical_converged','model_compatible', ...
    'rank_selection_resolved','selected_rank','active_rank','boundary_hit', ...
    'relative_residual','residual','nnls_residual','residual_target','exit_reason', ...
    'seed_failures','subproblem_failures','d_mean','d_mode'};
s=struct();
for j=1:numel(names),if isfield(f,names{j}),s.(names{j})=f.(names{j});end,end
end
