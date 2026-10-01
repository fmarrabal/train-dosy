function report=test_train_mf_signed(outputFile)
% Deterministic synthetic regression: truth is used only for evaluation.
% No experimental data or manuscript result is overwritten by this test.
here=fileparts(mfilename('fullpath'));addpath(fileparts(here));
rng(28092026,'twister');
D=logspace(log10(.02e-9),log10(5e-9),256)';
b=12e9*linspace(0,1,28)'.^2;K=exp(-b*D');
bt=12e9*linspace(.025,.975,13)'.^2;Kt=exp(-bt*D');
o=struct('r_max',4,'max_outer_iter',400,'verbose',true);
cases=cell(1,3);
for rank=1:3
    centers=[.22,.75,2.5]*1e-9;S=zeros(256,rank);
    for j=1:rank
        S(:,j)=exp(-.5*(log(D/centers(j))/.18).^2);S(:,j)=S(:,j)/sum(S(:,j));
    end
    % Separate signal regions plus frequencies containing genuine overlap.
    % Vary amplitudes within each region: frequencies share diffusion profiles.
    A=zeros(rank,6*rank);
    for j=1:rank,A(j,(j-1)*6+(1:6))=[1,.7,1.3,.8,1.1,.9];end
    if rank>1,A(:,end-2:end)=A(:,end-2:end)+.25;end
    truth=S*A;clean=K*truth;Y=clean+.001*randn(size(clean));original=Y;
    [X,~,fit]=TRAIn_DOSY_MF_Signed(Y,b,D,o);
    meanTruth=D'*truth./sum(truth,1);meanFit=D'*X./sum(X,1);
    meanError=max(abs(meanFit./meanTruth-1));
    testError=norm(Kt*(X-truth),'fro')/norm(Kt*truth,'fro');
    fprintf('Truth %d; selected %d; rankResolved %d; meanD %.5g; prediction %.5g; validation %s\n', ...
        rank,fit.selected_rank,fit.rank_selection_resolved,meanError,testError,mat2str(fit.validation_mean_loss,5));
    fprintf('Seed/subproblem failures %s / %s\n',mat2str(cellfun(@(f)f.seed_failures,fit.rank_trials)),mat2str(cellfun(@(f)f.subproblem_failures,fit.rank_trials)));
    assert(isequal(Y,original),'Input was changed.');
    assert(all(isfinite(X),'all')&&all(X>=0,'all'),'Distribution infeasible.');
    assert(all(diff(fit.residual_history)<=1e-11),'Objective increased.');
    assert(meanError<.08,'Mean D error exceeds 8%%.');
    assert(testError<.015,'Held-out clean attenuation error exceeds 1.5%%.');
    assert(fit.selected_rank==rank&&fit.rank_selection_resolved,'Predictive synthetic rank not recovered.');
    cases{rank}=struct('true_rank',rank,'selected_rank',fit.selected_rank, ...
        'status',fit.status,'max_mean_D_relative_error',meanError, ...
        'external_clean_relative_error',testError,'fit',fit);
    if rank==2
        [scaled,~,sf]=TRAIn_DOSY_MF_Signed(7*Y,b,D,o);
        meanScaled=D'*scaled./sum(scaled,1);
        scalePrediction=norm(K*(scaled/7-X),'fro')/norm(Y,'fro');
        scaleMean=max(abs(meanScaled./meanFit-1));
        assert(sf.selected_rank==fit.selected_rank&&scalePrediction<1e-4&&scaleMean<1e-3, ...
            'Global amplitude scaling changed predictions or D means.');
        limited=o;limited.n_components=1;
        [~,~,lf]=TRAIn_DOSY_MF_Signed(Y,b,D,limited);
        assert(~lf.success&&strcmp(lf.status,'train_mf_requires_review'),'Inadequate rank falsely approved.');
        permutation=randperm(size(Y,2));
        [permuted,~,pf]=TRAIn_DOSY_MF_Signed(Y(:,permutation),b,D,o);
        assert(pf.selected_rank==fit.selected_rank&&norm(K*(permuted-X(:,permutation)),'fro')/norm(Y,'fro')<1e-4);
        zeroTail=Y;zeroTail(end,end)=0;zeroTail(end-1,end)=-.001;
        [zx,~,zf]=TRAIn_DOSY_MF_Signed(zeroTail,b,D,o);
        assert(all(isfinite(zx),'all')&&zf.input_negative_count==nnz(zeroTail<0));
    end
end
% Reproduce the concrete minimum-subtraction failure with a surviving tail.
bTail=5.8e9*linspace(0,1,24)'.^2;ktail=exp(-bTail*D');
truthTail=exp(-.5*(log(D/.36e-9)/.14).^2);truthTail=truthTail/sum(truthTail);
yTail=ktail*truthTail+.0003*sin((1:24)'*1.7);
[new,~,nf]=TRAIn_DOSY_MF_Signed(yTail,bTail,D,struct('r_max',1));
[biased,bf]=trainmf.train(ktail,yTail-min(yTail),700,1.05);
actual=D'*truthTail;revised=D'*new/sum(new);old=D'*biased/sum(biased);
assert(abs(revised/actual-1)<.04&&abs(old/actual-1)>.1,'Minimum-subtraction bias regression failed.');
% Mask, physical-unit conversion and exact local/public wrapper behavior.
wide=[zeros(size(yTail)),yTail,zeros(size(yTail))];
[masked,axis,mi]=TRAIn_DOSY_MF(wide,bTail,[.02 5 256],struct('signal_mask',logical([0 1 0])));
assert(all(isnan(masked(:,[1 3])),'all')&&isequal(mi.process_idx,2));
% Equivalent log-grid construction differs by floating-point ulps. For an
% ill-conditioned ILT compare forward signal and D mean, not node identity.
assert(norm(axis*1e-9-D)<1e-20&&norm(ktail*(masked(:,2)-new))/norm(yTail)<1e-4);
assert(abs((axis'*1e-9*masked(:,2)/sum(masked(:,2)))/revised-1)<1e-3);
badGrid=false;try,TRAIn_DOSY_MF(wide,bTail,[.02 5 128]);catch e,badGrid=strcmp(e.identifier,'TRAInMF:Grid');end
assert(badGrid,'Sub-256 grid accepted.');
badPreprocessing=false;try,TRAIn_DOSY_MF(wide,bTail,[.02 5 256],struct('normalize_y',true));catch e,badPreprocessing=strcmp(e.identifier,'TRAInMF:LegacyOption');end
assert(badPreprocessing,'Legacy normalization silently accepted.');
report=struct('protocol',fit.protocol,'matlab_version',version,'cases',{cases}, ...
    'amplitude_scale',struct('prediction_relative_difference',scalePrediction,'mean_D_relative_difference',scaleMean), ...
    'minimum_subtraction',struct('true_mean_D',actual,'revised_mean_D',revised,'subtracted_mean_D',old, ...
    'tail_fraction',yTail(end)/yTail(1),'revised_status',nf.status,'subtracted_train_status',bf.exit_reason), ...
    'checks_passed',true,'scope','Synthetic regression, not experimental or chemical validation');
if nargin>0
    f=fopen(outputFile,'w');assert(f>=0);c=onCleanup(@()fclose(f));
    fprintf(f,'%s',jsonencode(report,PrettyPrint=true));
end
fprintf('PASS: 1/2/3 shared-profile cases, D bias, scale, permutation, signed/zero tails, rank cap, mask and option guards.\n');
end
