function report=audit_completed_outputs()
root=fileparts(mfilename('fullpath'));cfg=jsondecode(fileread(fullfile(root,'config.json')));
addpath(fullfile(root,'instrumented','mf_robust'));addpath(fullfile(root,'instrumented','mf_original','dosy_v4'));
checks=struct([]);paired=struct([]);idx=0;pairidx=0;initialization_equal=true;prediction_equal=true;objective_equal=true;feasible=true;pre_refinement_equal=true;monotone=true;
for rank=cfg.ranks(:)'
 for snr=cfg.snrs(:)'
  id=sprintf('rank%d_snr%d',rank,snr);z=load(fullfile(root,'inputs',[id '.mat']));baseline=[];caseFits=cell(1,numel(cfg.warm_sweeps));wi=0;
  for warm=cfg.warm_sweeps(:)'
   wi=wi+1;
   q=load(fullfile(root,'results',sprintf('%s_warm%02d.mat',id,warm)));
   if ~q.result.completed||~isempty(q.result.error),continue;end
   caseFits{wi}=q;
   p=dosyv4.prepare(z.Y(z.fit_rows,:),z.b(z.fit_rows),z.D,q.o);
   if isempty(baseline),baseline=q.info.starts;end
   for j=1:numel(q.info.starts)
    st=q.info.starts(j);
    initialization_equal=initialization_equal&&isequal(st.initial_S,baseline(j).initial_S)&&isequal(st.initial_A,baseline(j).initial_A);
    [v,~,k]=mfrobust.evaluate(p.K,p.Y,st.S,st.A/p.scale,p.L,p.options);
    objective_equal=objective_equal&&abs(v-st.objective)<=1e-12*max(1,abs(v))&&abs(k.relative-st.kkt.relative)<=1e-11;
    feasible=feasible&&min(st.S,[],'all')>=-1e-10&&min(st.A,[],'all')>=0&&max(abs(sum(st.S)-1))<1e-8;
    pre=st.pre_refinement;[pv,~,pk]=mfrobust.evaluate(p.K,p.Y,pre.S,pre.A/p.scale,p.L,p.options);
    pre_refinement_equal=pre_refinement_equal&&abs(pv-pre.objective)<=1e-12*max(1,abs(pv))&&abs(pk.relative-pre.kkt.relative)<=1e-11 ...
      &&max(abs(p.K*pre.S*pre.A-pre.prediction),[],'all')<=1e-12;
    monotone=monotone&&all(diff(st.history)<=1e-10*max(1,st.history(1:end-1)));
   end
   pred=exp(-z.b(z.fit_rows)*z.D')*q.X;
   prediction_equal=prediction_equal&&max(abs(pred-q.info.full_prediction),[],'all')<=1e-12;
   idx=idx+1;record=struct('id',id,'warm_cap',warm,'history_monotone_at_original_tolerance', ...
    all(diff(q.info.history)<=1e-10*max(1,q.info.history(1:end-1))), ...
    'selected_kkt',q.info.kkt.relative);
   if idx==1,checks=record;else,checks(idx)=record;end
  end
  if any(cellfun(@isempty,caseFits)),continue;end
  q8=caseFits{find(cfg.warm_sweeps==8,1)};q16=caseFits{find(cfg.warm_sweeps==16,1)};
  pred8=exp(-z.b*z.D')*q8.X;pred16=exp(-z.b*z.D')*q16.X;
  truth=z.Yclean;best=min(cellfun(@(q)q.info.objective,caseFits));
  pairidx=pairidx+1;pr=struct('id',id,'rank',rank,'snr',snr, ...
   'objective8',q8.info.objective,'objective16',q16.info.objective,'best_attained_objective',best, ...
   'relative_objective_difference_16_minus_8',(q16.info.objective-q8.info.objective)/max(abs(q8.info.objective),realmin), ...
   'prediction_difference_all_rows_rmse_sigma',sqrt(mean((pred8-pred16).^2,'all'))/z.sigma, ...
   'prediction_difference_external_rmse_sigma',sqrt(mean((pred8(z.test_rows,:)-pred16(z.test_rows,:)).^2,'all'))/z.sigma, ...
   'prediction_difference_relative_fro',norm(pred8-pred16,'fro')/norm(truth,'fro'), ...
   'final_kkt8',q8.info.kkt.relative,'final_kkt16',q16.info.kkt.relative, ...
   'clean_external_rmse8_sigma',q8.result.test_clean_rmse_sigma,'clean_external_rmse16_sigma',q16.result.test_clean_rmse_sigma);
  if pairidx==1,paired=pr;else,paired(pairidx)=pr;end
 end
end
report=struct('fits_checked',idx,'identical_initializations_across_caps',initialization_equal, ...
 'stored_predictions_recomputed',prediction_equal,'stored_objectives_and_KKT_recomputed',objective_equal, ...
 'stored_pre_refinement_recomputed',pre_refinement_equal,'all_returned_starts_feasible',feasible, ...
 'all_start_objective_histories_monotone_at_original_tolerance',monotone,'records',checks,'warm8_vs16',paired);
report.environment=struct('MATLAB',version,'Optimization_Toolbox',ver('optim'),'computer',computer);
report.passed=idx==36&&initialization_equal&&prediction_equal&&objective_equal&&feasible&&pre_refinement_equal&&monotone;
fid=fopen(fullfile(root,'completed_output_audit.json'),'w');fprintf(fid,'%s',jsonencode(report,PrettyPrint=true));fclose(fid);
assert(report.passed,'MFRevision:Audit','Inspect completed output audit.');
fprintf('PASS 36 saved fits: identical starts, objective/KKT/prediction replay, feasibility.\n');
end
