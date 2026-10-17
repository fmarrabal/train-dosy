function result=test_runtime_curvature()
root=fileparts(mfilename('fullpath'));addpath(fullfile(root,'instrumented','mf_robust'));
addpath(fullfile(root,'instrumented','mf_original','dosy_v4'));
D=logspace(log10(.1e-9),log10(15e-9),256)';b=linspace(0,2.2e9,8)';
A=[1 .2 .8;.2 1 .4];K=exp(-b*D');S=zeros(256,2);S([80 160],:)=eye(2);
o=struct('units','SI','n_components',2,'lambda_S',1e-8,'lambda_A',1e-8,'solver_max_iter',2000);
p=dosyv4.prepare(K*S*A,b,D,o);design=mfrobust.curvature_design(K,p.L,D,p.quadrature,1e-10);
independent=mfrobust.curvature_report(A,design);assert(independent.supported);
duplicate=mfrobust.curvature_report([1 .5 .2;1 .5 .2],design);assert(~duplicate.supported&&duplicate.numeric_row_rank==1);
constant=mfrobust.curvature_report(A,mfrobust.curvature_design(ones(size(K)),p.L,D,p.quadrature,1e-10));assert(~constant.supported);
zero=mfrobust.curvature_report(zeros(size(A)),design);assert(~zero.supported&&zero.numeric_row_rank==0);
opt=p.options;opt.runtime_design=design;opt.runtime_fallback=true;
dupA=[1 .5 .2;1 .5 .2];before=mfrobust.evaluate(K,K*S*dupA,S,dupA,p.L,opt);
[updated,meta]=mfrobust.simplex_block(K,K*S*dupA,dupA,p.L,opt.lambda_S,S,opt);
after=mfrobust.evaluate(K,K*S*dupA,updated,dupA,p.L,opt);
assert(meta.curvature_fallback&&meta.curvature.fallback_used&&~meta.curvature.supported);
assert(min(updated,[],'all')>=-1e-10&&max(abs(sum(updated)-1))<1e-8&&after<=before+1e-10);
result=struct('passed',true,'tests',5,'independent',independent,'duplicate',duplicate,'constant_kernel',constant,'zero_amplitudes',zero, ...
 'duplicate_runtime_fallback',meta,'before_objective',before,'after_objective',after,'matlab_version',version);
fid=fopen(fullfile(root,'runtime_curvature_tests.json'),'w');fprintf(fid,'%s',jsonencode(result,PrettyPrint=true));fclose(fid);
fprintf('PASS 5 runtime-curvature tests, including actual duplicated-row fallback.\n');
end
