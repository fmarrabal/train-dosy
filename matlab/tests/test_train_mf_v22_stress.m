function report=test_train_mf_v22_stress(outputFile)
% Independent stress checks; repeated nulls do not calibrate experimental FPR.
here=fileparts(mfilename('fullpath'));addpath(fileparts(here));
previous=rng;restore=onCleanup(@()rng(previous)); %#ok<NASGU>
D=logspace(log10(.02e-9),log10(5e-9),256)';
b=12e9*linspace(0,1,24)'.^2;K=exp(-b*D');
rng(20261003,'twister');Y=exp(-b*.7e-9)*[1,.7,1.2,.5]+.001*randn(24,4);
val=false(24,1);val(3:4:23)=true;
o=struct('r_max',2,'validation_rows',val);
[~,~,a]=TRAIn_DOSY_MF_Signed(Y,b,D,o);
changed=Y;changed(val,:)=changed(val,:)*1e4;
[~,~,c]=TRAIn_DOSY_MF_Signed(changed,b,D,o);
originalResidual=cellfun(@(t)t.residual,a.rank_trials);
changedResidual=cellfun(@(t)t.residual,c.rank_trials);
assert(isequal(originalResidual,changedResidual)&&a.candidate_fit_scale==c.candidate_fit_scale, ...
    'Validation values changed a candidate fit.');

o=struct('n_components',1,'sigma',.001);
[reference,~,f]=TRAIn_DOSY_MF_Signed(Y,b,D,o);
unitScales=[1e-150,1e150];scaleDifferences=zeros(size(unitScales));
for j=1:numel(unitScales)
    units=unitScales(j);scaledOptions=o;scaledOptions.sigma=o.sigma*units;
    [scaled,~,s]=TRAIn_DOSY_MF_Signed(Y*units,b,D,scaledOptions);
    scaleDifferences(j)=norm(K*(scaled/units-reference),'fro')/norm(Y,'fro');
    assert(all(isfinite(scaled),'all')&&scaleDifferences(j)<1e-4&&s.model_compatible==f.model_compatible);
end

nullTrials=cell(1,20);
for j=1:numel(nullTrials)
    rng(20261020+j,'twister');
    [~,~,f]=TRAIn_DOSY_MF_Signed(.01*randn(24,6),b,D,struct('r_max',3,'sigma',.01));
    nullTrials{j}=struct('seed',20261020+j,'selected_rank',f.selected_rank, ...
        'success',f.success,'model_compatible',f.model_compatible, ...
        'signal_z',f.validation_signal_z,'signal_threshold',f.validation_signal_threshold);
end

% These formulas are independent algebraic/finite-difference checks, not an
% assertion of joint global optimality for the nonconvex factorization.
rng(21);k=exp(-linspace(0,4,13)'*logspace(-1,1,9));
S=rand(9,2);S=S./sum(S,1);A=rand(2,5);y=k*S*A+.02*randn(13,5);
R=y-k*S*A+(k*S(:,1))*A(1,:);target=R*A(1,:)'/(A(1,:)*A(1,:)');
h=rand(9,1);full=2*k'*(k*h*A(1,:)-R)*A(1,:)';
conditional=2*(A(1,:)*A(1,:)')*k'*(k*h-target);
eta=.1+rand(9,1);v=randn(9,1);delta=1e-6;
objective=@(t)norm(k*(t.^2)-y(:,1))^2;
gradient=4*eta.*(k'*(k*(eta.^2)-y(:,1)));
fd=(objective(eta+delta*v)-objective(eta-delta*v))/(2*delta);
J=2*k.*eta';hv=8*eta.*(k'*(k*(eta.*v)));
math=struct('conditional_gradient_relative_error',norm(full-conditional)/norm(full), ...
    'trust_gradient_relative_error',abs(fd-gradient'*v)/max(1,abs(fd)), ...
    'gauss_newton_product_relative_error',norm(hv-2*J'*J*v)/norm(hv));
assert(math.conditional_gradient_relative_error<1e-12&&math.trust_gradient_relative_error<1e-6&& ...
    math.gauss_newton_product_relative_error<1e-12);
report=struct('protocol',a.protocol,'matlab_version',version,'checks_passed',true, ...
    'training_residuals_before_validation_change',originalResidual, ...
    'training_residuals_after_validation_change',changedResidual, ...
    'intensity_unit_scales',unitScales,'intensity_prediction_difference',scaleDifferences, ...
    'null_trials',{nullTrials},'null_nonzero_selections',sum(cellfun(@(t)t.selected_rank>0,nullTrials)), ...
    'math',math,'scope','Synthetic stress checks; no experimental noise calibration or species identification.');
fout=fopen(outputFile,'w');assert(fout>=0);closeFile=onCleanup(@()fclose(fout)); %#ok<NASGU>
fprintf(fout,'%s',jsonencode(report,PrettyPrint=true));
fprintf('PASS: training isolation, extreme intensity units, algebra; %d/20 nonzero null selections.\n',report.null_nonzero_selections);
end
