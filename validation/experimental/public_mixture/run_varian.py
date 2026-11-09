from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,traceback,argparse
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent/'prop_pent_suc'
sys.path.insert(0,str(HERE.parent.parent))
import run_experimental as exp
from score_nnls_cv import fit_score_nnls_cv
METHODS=['RAI-S','DOME-S','SCORE-CV','SCORE-NNLS-CV','LOCODOSY-SCORE']

def run(method):
    target=HERE/'results'/f'{method}.json';target.parent.mkdir(exist_ok=True)
    if target.exists():print('SKIP',method,flush=True);return
    d=np.load(HERE/'input.npz');test=d['outer_test'];train=~test
    b=d['b'];y=d['Y'];sigma=float(d['sigma']);ppm=d['ppm'];start=time.perf_counter();local_calls=[]
    try:
        if method in ['RAI-S','DOME-S']:
            r=exp.support.fit_support(b[train],y[train],sigma,ppm,(.01e-9,15e-9),method=method.split('-')[0],config=exp.support.Config(max_components=4))
        elif method=='SCORE-CV':r=exp.score_cv(b[train],y[train],sigma)
        elif method=='SCORE-NNLS-CV':r=fit_score_nnls_cv(b[train],y[train],sigma)
        else:
            original=exp.loco.score
            def audited_score(*args,**kwargs):
                out=original(*args,**kwargs)
                local_calls.append({k:out.get(k) for k in ['D','success','evaluations','objective','message']})
                return out
            exp.loco.score=audited_score
            try:r=exp.loco.fit(b[train],y[train],ppm,windows=d['windows'],config=exp.loco.Config(method='score',max_components=4,diffusion_bins=256,max_diffusion=15e-9))
            finally:exp.loco.score=original
            r['A']=r.pop('amplitudes')
        D=r['D'];A=r['A'];pred=np.exp(-b[:,None]*D)@A;err=pred[test]-y[test]
        rms=float(np.sqrt(np.mean(y[train]**2)));k=np.exp(-b[train,None]*D)
        cond=float(np.linalg.cond(k)) if k.size else 0.;expansion=float(abs(A).max(initial=0)/abs(y).max())
        excluded=['A','D','prediction','groups','support','train_only_native','mass_grid','legacy_DOSY','windows','window_points']
        diag={k:v for k,v in r.items() if k not in excluded}
        record=dict(method=method,status='completed',seconds=time.perf_counter()-start,numerical_success=r.get('success'),
            selected_rank=len(D),D_SI=D,test_nrmse=float(np.sqrt(np.mean(err**2))/rms),
            test_rmse_over_sigma=float(np.sqrt(np.mean(err**2))/sigma),training_nrmse=float(np.sqrt(np.mean((pred[train]-y[train])**2))/rms),
            A_negative_mass_ratio=float(np.maximum(-A,0).sum()/max(abs(A).sum(),1e-300)),global_kernel_condition=cond,
            amplitude_expansion=expansion,unstable_global_amplitudes=bool(method!='LOCODOSY-SCORE' and (cond>1e8 or expansion>1e4)),
            atomic_model_mismatch=diag.get('atomic_model_mismatch_flag'),search_cap_reached=bool(method!='LOCODOSY-SCORE' and len(D)==4),
            points=y.shape[1],diagnostics=diag,local_solver_calls=local_calls,
            local_nonconvergence=sum(c['success'] is False for c in local_calls),
            source_input_sha256=exp.sha(HERE/'input.npz'))
        np.savez_compressed(target.with_suffix('.npz'),D=D,A=A,prediction=pred,b=b,ppm=ppm,outer_test=test,residual_test=err)
        exp.save_json(target,record)
        print('DONE',method,'rank',len(D),'NRMSE',100*record['test_nrmse'],'success',record['numerical_success'],'seconds',record['seconds'],flush=True)
    except Exception as e:
        exp.save_json(target,dict(method=method,status='failed',error=repr(e),traceback=traceback.format_exc()))
        print('FAILED',method,repr(e),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--methods',nargs='*',default=METHODS);args=parser.parse_args()
    # Already verified against native NNLS on the resin benchmark and exactly
    # equal end-to-end; this changes no objective or initialization.
    exp.components._profile=exp.fast_profile
    with threadpool_limits(limits=1):
        for method in args.methods:run(method)
