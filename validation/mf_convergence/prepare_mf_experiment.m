function prepare_mf_experiment()
root=fileparts(mfilename('fullpath'));
assert(~isfile(fullfile(root,'config.json')),'MFRevision:AlreadyPrepared','Do not overwrite frozen experiment configuration.');
mkdir(fullfile(root,'inputs'));mkdir(fullfile(root,'results'));mkdir(fullfile(root,'logs'));
config=struct('created_utc',char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd HH:mm:ss')), ...
 'scope','Controlled synthetic convergence study; conditional known factor count, not chemical identification or rank-selection validation', ...
 'warm_sweeps',[1 4 8 16],'sqpcap',250,'n_starts',2,'kkt_tol',1e-6, ...
 'objective_tol',1e-9,'solver_tol',1e-10,'solver_max_iter',2000, ...
 'curvature_tolerance',1e-10,'fallback_on_unresolved',true,'lambda_S',1e-8,'lambda_A',1e-8, ...
 'bins',256,'acquisitions',32,'channels',24,'ranks',[1 2 3],'snrs',[50 100 200], ...
 'noise_seed_base',20261017,'solver_seed_base',730000,'replicates',1, ...
 'comparison','Same inputs, initialization seeds, SQP budget, tolerances and objective across warm budgets. Runtime condition failure routes S to interior-point, without altering objective.', ...
 'source_version','Frozen regularized MATLAB reference 5.1; not signed-v2.2; archived illustration itself was 5.0', ...
 'runtime_scope','Condition checked before every S block and at each accepted SQP iteration; unresolved status never proves nonuniqueness of a boundary solution.');
fid=fopen(fullfile(root,'config.json'),'w');fprintf(fid,'%s',jsonencode(config,PrettyPrint=true));fclose(fid);
D=logspace(log10(.1e-9),log10(15e-9),256)';b=2.2e9*((0:31)'/31).^1.7;
fit_rows=mod((0:31)',4)~=3;test_rows=~fit_rows;ppm=linspace(7.2,8.2,24);
x=log(D);dx=diff(x);w=[dx(1)/2;(dx(1:end-1)+dx(2:end))/2;dx(end)/2];
phi=[exp(-.5*((ppm-7.36)/.12).^2);exp(-.5*((ppm-7.73)/.12).^2);exp(-.5*((ppm-8.05)/.12).^2)];
mix=[1 .40 .15;.15 1 .35;.35 .25 1];centers=[.7 2.3 7.1]*1e-9;
cases=struct([]);idx=0;
for rank=config.ranks
 Strue=zeros(numel(D),rank);
 for j=1:rank,s=w.*exp(-.5*((x-log(centers(j)))/.14).^2);Strue(:,j)=s/sum(s);end
 Atrue=mix(1:rank,:)*phi;
 Atrue=Atrue/max(sum(Atrue,1));Yclean=exp(-b*D')*Strue*Atrue;
 for snr=config.snrs
  idx=idx+1;seed=config.noise_seed_base+1000*rank+snr;rng(seed,'twister');sigma=1/snr;
  Y=Yclean+sigma*randn(size(Yclean));id=sprintf('rank%d_snr%d',rank,snr);solver_seed=config.solver_seed_base+idx;
  save(fullfile(root,'inputs',[id '.mat']),'Y','Yclean','b','D','Strue','Atrue','rank','snr','sigma','seed','solver_seed','fit_rows','test_rows','ppm','-v7');
  caseRecord=struct('id',id,'rank',rank,'snr',snr,'noise_seed',seed,'solver_seed',solver_seed, ...
   'minimum_unattenuated_channel_signal',min(sum(Atrue,1)), 'spectral_cosines',(Atrue./vecnorm(Atrue,2,2))*(Atrue./vecnorm(Atrue,2,2))');
  if idx==1,cases=caseRecord;else,cases(idx)=caseRecord;end
 end
end
fid=fopen(fullfile(root,'cases.json'),'w');fprintf(fid,'%s',jsonencode(cases,PrettyPrint=true));fclose(fid);
fprintf('Saved configuration and all 9 synthetic inputs before any inversion.\n');
end
