"""Explicit positive-amplitude variant of GNAT SCORE, with acquisition CV.

Not the signed SCORE default. Objective/minimizer/stopping budget are inherited
from the attributed native translation; only conditional amplitudes use NNLS.
"""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,time,argparse,json,traceback
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_experimental as exp
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
original_amplitudes=exp.loco.amplitudes

def batched_amplitudes(k,y,nonnegative=False):
    return exp.amplitudes_exact(k,y) if nonnegative else original_amplitudes(k,y,False)

def validate():
    rng=np.random.default_rng(17102027);checks=[]
    for rank in range(1,5):
        b=np.linspace(0,5,18);D=np.geomspace(.15,2.5,rank);k=np.exp(-b[:,None]*D)
        y=k@rng.uniform(0,1,(rank,37))+.05*rng.normal(size=(18,37));y[:,-4:]-=2
        a=original_amplitudes(k,y,True);c=batched_amplitudes(k,y,True)
        rel=float(np.linalg.norm(a-c)/max(np.linalg.norm(a),1e-300))
        assert rel<1e-8
        checks.append(dict(rank=rank,amplitude_relative_error=rel))
    exp.save_json(HERE/'score_nnls_batch_equivalence.json',dict(checks=checks,passed=True,
        native_source_sha256=exp.sha(exp.LOCOPATH),batch_source_sha256=exp.sha(sys.modules['dome.solver'].__file__)))

def fit_score_nnls_cv(b,y,sigma,max_components=4):
    """Return D,A,success and CV diagnostics; b SI, signed Y unchanged."""
    b=np.asarray(b,float);y=np.asarray(y,float)
    val=np.zeros(len(b),bool);val[2:-1:4]=True;train=~val
    cfg=exp.loco.Config(method='score',max_components=max_components,auto_reduce=False,
        max_evaluations=10000,nonnegative=True)
    previous=exp.loco.amplitudes;exp.loco.amplitudes=batched_amplitudes
    try:
        candidates=[]
        for rank in range(1,max_components+1):
            r=exp.loco.score(b[train],y[train],rank,cfg)
            pred=np.exp(-b[val,None]*r['D'])@r['amplitudes']
            candidates.append(dict(rank=rank,D=r['D'],prediction=pred,
                sse=float(np.sum(((pred-y[val])/sigma)**2)),
                success=bool(r['success'] and np.all(r['D']>0)),evaluations=r['evaluations']))
        good=[i for i,c in enumerate(candidates) if c['success']]
        if not good:raise RuntimeError('No converged positive-rate SCORE-NNLS candidate')
        best=min(good,key=lambda i:candidates[i]['sse'])
        eligible=[i for i in good if candidates[i]['sse']<=candidates[best]['sse']+
            2*np.linalg.norm(candidates[i]['prediction']-candidates[best]['prediction'])/sigma+1e-10]
        chosen=min(eligible,key=lambda i:(candidates[i]['rank'],candidates[i]['sse']))
        out=exp.loco.score(b,y,candidates[chosen]['rank'],cfg)
        out.update(A=out.pop('amplitudes'),selected_rank=candidates[chosen]['rank'],
            candidate_diagnostics=[{k:v for k,v in c.items() if k!='prediction'} for c in candidates],
            variant='SCORE with NNLS amplitudes, same native initialization and Nelder-Mead budgets; common CV wrapper. Not signed upstream default.')
        return out
    finally:exp.loco.amplitudes=previous

def run(identity):
    method='SCORE-NNLS-CV';target=HERE/'results'/f'{identity}__{method}.json'
    if target.exists():print('SKIP',identity,method,flush=True);return
    d=np.load(HERE/'inputs'/f'{identity}.npz');test=d['outer_test'];train=~test
    b=d['b'];y=d['Y'];sigma=float(d['sigma']);start=time.perf_counter()
    try:
        r=fit_score_nnls_cv(b[train],y[train],sigma)
        D=r['D'];A=r['A'];pred=np.exp(-b[:,None]*D)@A
        err=pred[test]-y[test];rms=np.sqrt(np.mean(y[train]**2))
        diag={k:v for k,v in r.items() if k not in ('A','D')}
        out=dict(id=identity,method=method,seconds=time.perf_counter()-start,status='completed',
            numerical_success=r['success'],selected_rank=len(D),rank_interpretation='global atoms, not chemical truth',
            D_SI=D,A_negative_mass_ratio=float(np.maximum(-A,0).sum()/max(abs(A).sum(),1e-300)),
            test_rmse=float(np.sqrt(np.mean(err**2))),test_nrmse=float(np.sqrt(np.mean(err**2))/rms),
            test_rmse_over_sigma=float(np.sqrt(np.mean(err**2))/sigma),test_bias_over_sigma=float(err.mean()/sigma),
            training_nrmse=float(np.sqrt(np.mean((pred[train]-y[train])**2))/rms),points=y.shape[1],diagnostics=diag)
        np.savez_compressed(target.with_suffix('.npz'),D=D,A=A,prediction=pred,ppm=d['ppm'],b=b,outer_test=test,
            support=np.ones_like(A,dtype=bool),residual_test=err)
        exp.save_json(target,out)
        print('DONE',identity,method,'sec',round(out['seconds'],1),'rank',len(D),'nrmse',out['test_nrmse'],'success',r['success'],flush=True)
    except Exception as e:
        exp.save_json(target,dict(id=identity,method=method,status='failed',error=repr(e),seconds=time.perf_counter()-start,traceback=traceback.format_exc()))
        print('FAILED',identity,repr(e),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--ids',nargs='*');args=ap.parse_args()
    with threadpool_limits(limits=1):
        validate()
        for identity in args.ids or [p.stem for p in sorted((HERE/'inputs').glob('*.npz'))]:run(identity)
