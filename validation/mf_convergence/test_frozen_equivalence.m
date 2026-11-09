function result=test_frozen_equivalence(id)
% Run in a fresh MATLAB process to avoid loading instrumented package members.
root=fileparts(mfilename('fullpath'));addpath(fullfile(root,'source_frozen','mf_robust'));
z=load(fullfile(root,'inputs',[id '.mat']));q=load(fullfile(root,'results',[id '_warm08.mat']));
assert(q.result.completed&&sum([q.result.starts.curvature_fallbacks])==0);
t=tic;[X,~,info]=DOSY_MF_Robust(z.Y(z.fit_rows,:),z.b(z.fit_rows),z.D,q.o);
result=struct('id',id,'relative_prediction_difference',norm(exp(-z.b*z.D')*(X-q.X),'fro')/max(norm(exp(-z.b*z.D')*X,'fro'),realmin), ...
 'relative_mass_difference',norm(X-q.X,'fro')/max(norm(X,'fro'),realmin), ...
 'objective_difference',abs(info.objective-q.info.objective),'kkt_difference',abs(info.kkt.relative-q.info.kkt.relative), ...
 'wall_seconds',toc(t));
result.passed=result.relative_prediction_difference<1e-12&&result.objective_difference<1e-12;
assert(result.passed,'MFRevision:Regression','Instrumentation changed numerical output on the fixed full-rank regression.');
save(fullfile(root,['frozen_equivalence_' id '.mat']),'X','info','result','-v7.3');
fid=fopen(fullfile(root,['frozen_equivalence_' id '.json']),'w');fprintf(fid,'%s',jsonencode(result,PrettyPrint=true));fclose(fid);
fprintf('PASS frozen equivalence %s: prediction relative difference %.3g\n',id,result.relative_prediction_difference);
end
