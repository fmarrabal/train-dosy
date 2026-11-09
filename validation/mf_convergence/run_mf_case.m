function result=run_mf_case(id,warm)
root=fileparts(mfilename('fullpath'));addpath(fullfile(root,'instrumented','mf_robust'));
config=jsondecode(fileread(fullfile(root,'config.json')));z=load(fullfile(root,'inputs',[id '.mat']));
outfile=fullfile(root,'results',sprintf('%s_warm%02d.mat',id,warm));
if isfile(outfile),q=load(outfile,'result');result=q.result;fprintf('RETAIN %s warm%d\n',id,warm);return;end
o=struct('units','SI','n_components',z.rank,'n_starts',config.n_starts,'seed',z.solver_seed, ...
 'max_outer_iter',config.sqpcap,'objective_tol',config.objective_tol,'kkt_tol',config.kkt_tol, ...
 'solver_tol',config.solver_tol,'solver_max_iter',config.solver_max_iter, ...
 'lambda_S',config.lambda_S,'lambda_A',config.lambda_A,'verbose',true);
controls=struct('warm_sweeps',warm,'curvature_tolerance',config.curvature_tolerance, ...
 'fallback_on_unresolved',config.fallback_on_unresolved);
timer=tic;cpu=cputime;info=[];X=[];starts=struct([]);
result=struct('id',id,'rank',z.rank,'snr',z.snr,'warm_sweeps',warm,'completed',false,'numerical_success',false,'error','');
try
 [X,D,info]=DOSY_MF_Runtime(z.Y(z.fit_rows,:),z.b(z.fit_rows),z.D,o,controls);
 result.completed=true;result.numerical_success=info.kkt.relative<=config.kkt_tol;
 result.objective=info.objective;result.final_kkt=info.kkt.relative;result.status=info.status;
 pred=exp(-z.b(z.test_rows)*z.D')*X;
 result.test_clean_rmse_sigma=sqrt(mean((pred-z.Yclean(z.test_rows,:)).^2,'all'))/z.sigma;
 result.test_observed_rmse_sigma=sqrt(mean((pred-z.Y(z.test_rows,:)).^2,'all'))/z.sigma;
 result.training_rmse_sigma=sqrt(mean(info.full_residual.^2,'all'))/z.sigma;
 for j=1:numel(info.starts)
  st=info.starts(j);ref=st.profile_refinement;ri=0;rf=NaN;rv=[];if isfield(ref,'iterations'),ri=ref.iterations;rf=ref.exitflag;rv=ref.curvature_log;end
  allcurv=[st.curvature_log(:);rv(:)];
  startRecord=struct('start',j,'warm_iterations',st.iterations,'pre_kkt',st.pre_refinement.kkt.relative, ...
   'pre_objective',st.pre_refinement.objective,'post_kkt',st.kkt.relative,'post_objective',st.objective, ...
   'sqp_iterations',ri,'sqp_exitflag',rf,'numerical_success',st.kkt.relative<=config.kkt_tol, ...
   'runtime_checks',numel(allcurv),'unresolved_runtime_checks',sum(~[allcurv.supported]), ...
   'curvature_fallbacks',sum([allcurv.fallback_used]),'S_solver_nonpositive_flags',st.S_solver_nonpositive_flags, ...
   'minimum_amplitude_ratio',min([allcurv.amplitude_ratio]),'kernel_ratio',st.final_curvature.kernel_ratio, ...
   'final_condition_supported',st.final_curvature.supported);
  if j==1,starts=startRecord;else,starts(j)=startRecord;end
 end
 result.starts=starts;result.objective_history_monotone=all(diff(info.history)<=1e-10*max(1,info.objective));
catch ex
 result.error=getReport(ex,'extended','hyperlinks','off');fprintf(2,'%s\n',result.error);
end
result.wall_seconds=toc(timer);result.cpu_seconds=cputime-cpu;result.matlab_version=version;
save(outfile,'result','info','X','o','controls','-v7.3');
fid=fopen(strrep(outfile,'.mat','.json'),'w');fprintf(fid,'%s',jsonencode(result,PrettyPrint=true));fclose(fid);
fprintf('CASE %s warm%d success%d elapsed%.1f CPU%.1f\n',id,warm,result.numerical_success,result.wall_seconds,result.cpu_seconds);
end
