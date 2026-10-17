"""Apply narrowly scoped instrumentation to the preserved MATLAB reference copy."""
from pathlib import Path
import argparse, hashlib, json, shutil
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--destination',type=Path,default=root/'instrumented');args=parser.parse_args()
destination=args.destination.resolve();assert destination.is_relative_to(root.resolve())
shutil.copytree(root/'source_frozen',destination,dirs_exist_ok=True)
folder=destination/'mf_robust'
if destination != (root/'instrumented').resolve():
    for name in ['curvature_design.m','curvature_report.m']:
        shutil.copy2(root/'instrumented/mf_robust/+mfrobust'/name,folder/'+mfrobust'/name)
def replace_once(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)
p=folder/'DOSY_MF_Robust.m';s=p.read_text()
s=replace_once(s,'function [X,D,info]=DOSY_MF_Robust(Y,x,D,options)', 'function [X,D,info]=DOSY_MF_Runtime(Y,x,D,options,controls)')
s=replace_once(s,'if nargin<4,options=struct();end',"if nargin<4,options=struct();end\nif nargin<5,controls=struct('warm_sweeps',8,'curvature_tolerance',1e-10,'fallback_on_unresolved',true);end\nassert(isscalar(controls.warm_sweeps)&&controls.warm_sweeps>=1&&controls.warm_sweeps==floor(controls.warm_sweeps));")
s=replace_once(s,'t=tic;[nd,r,nf]', "o.runtime_design=mfrobust.curvature_design(K,L,D,p.quadrature,controls.curvature_tolerance);\no.runtime_fallback=controls.fallback_on_unresolved;\nt=tic;[nd,r,nf]")
s=replace_once(s,"[val,parts]=mfrobust.evaluate(K,V,S,A,L,o);history=val;", "initial_S=S;initial_A=A*p.scale;curvature_log=struct([]);\n    [val,parts]=mfrobust.evaluate(K,V,S,A,L,o);history=val;")
s=replace_once(s,'for iteration=1:min(8,o.max_outer_iter)','for iteration=1:min(controls.warm_sweeps,o.max_outer_iter)')
s=replace_once(s,'subfail=subfail+(sm.exitflag<=0);',"subfail=subfail+(sm.exitflag<=0);\n        cd=sm.curvature;cd.stage='before_S_block';cd.iteration=iteration;if isempty(curvature_log),curvature_log=cd;else,curvature_log(end+1)=cd;end")
s=replace_once(s,"refinement=struct();", "pre_refinement=struct('objective',val,'kkt',kkt,'S',S,'A',A*p.scale,'prediction',K*S*A*p.scale);\n    refinement=struct();")
s=replace_once(s,"'profile_refinement',refinement);", "'profile_refinement',refinement,'initial_S',initial_S,'initial_A',initial_A,'pre_refinement',pre_refinement, ...\n        'curvature_log',curvature_log,'final_curvature',mfrobust.curvature_report(A,o.runtime_design));")
s=replace_once(s,"'version','5.1-feasible-stationary-selection'", "'version','5.1-instrumented-warm-budget-curvature'")
s=replace_once(s,"info.identifiability=", "info.runtime_controls=controls;\ninfo.identifiability=")
(folder/'DOSY_MF_Runtime.m').write_text(s,encoding='utf-8')
# Keep the original copied entry point unchanged as an explicit reference.
p=folder/'+mfrobust/simplex_block.m';s=p.read_text()
s=replace_once(s,"[x,~,~,flag,out]=lsqlin(C,rhs,[],[],eq,ones(r,1),zeros(nd*r,1),[],x0,opts);", "curvature=mfrobust.curvature_report(A,o.runtime_design);\nforced_interior=o.runtime_fallback&&~curvature.supported;\ncurvature.fallback_used=forced_interior;\nxstart=x0;if forced_interior,opts.Algorithm='interior-point';xstart=[];end\n[x,~,~,flag,out]=lsqlin(C,rhs,[],[],eq,ones(r,1),zeros(nd*r,1),[],xstart,opts);")
s=replace_once(s,"'rejected_infeasible',rejected_infeasible);", "'rejected_infeasible',rejected_infeasible,'curvature',curvature,'curvature_fallback',forced_interior);")
p.write_text(s,encoding='utf-8')
p=folder/'+mfrobust/profile_refine.m';s=p.read_text()
s=replace_once(s,'warm=A;values=[];evaluations=0;', 'warm=A;values=[];evaluations=0;curvature_log=struct([]);')
s=replace_once(s,"'first_order_optimality',out.firstorderopt,'accepted',accepted);", "'first_order_optimality',out.firstorderopt,'accepted',accepted,'curvature_log',curvature_log);")
s=replace_once(s,'function stop=trace(~,v,state)', 'function stop=trace(ztrace,v,state)')
s=replace_once(s,"values(end+1)=v.fval;", "values(end+1)=v.fval;\n            traceS=reshape(ztrace,nd,r);traceA=mfrobust.nnls_batch(K*traceS,Y,o.lambda_A,[]);\n            cd=mfrobust.curvature_report(traceA,o.runtime_design);cd.stage='SQP_iteration';cd.iteration=v.iteration;if isempty(curvature_log),curvature_log=cd;else,curvature_log(end+1)=cd;end")
p.write_text(s,encoding='utf-8')
files=[]
for subtree in [root/'source_frozen',destination]:
 for p in sorted(subtree.rglob('*.m')):
  files.append({'file':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(destination/'source_hashes.json').write_text(json.dumps(files,indent=2))
print('Instrumented isolated copy; frozen original and signed-v2.2 untouched.')
