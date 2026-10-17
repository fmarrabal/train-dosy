"""Paired 2x2 screening x predictive-order selection, frozen atomic solvers."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,hashlib,time,concurrent.futures,argparse
from datetime import datetime,timezone
import numpy as np

BASE=Path(__file__).resolve().parent;OUT=BASE/'factorial';REPO=BASE.parents[3]
sys.path.insert(0,str(REPO/'src'));sys.path.insert(0,str(REPO/'benchmarks'))
from train_dosy._frozen.support import native_path,refine,screen_support,groups_from_ppm
from train_dosy.inversion import clean_json
from synthetic import generate
from evaluate import metrics
INDICES=[i for i in range(48) if i%8<6]
BOUNDS=(.1e-9,15e-9)
HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(clean_json(v),indent=2,allow_nan=False),encoding='utf-8')

def choose(cs,sigma):
    good=[c for c in cs if c['success'] and c['kkt']<=1e-7]
    if not good:raise RuntimeError('No numerically eligible candidate')
    best=min(good,key=lambda c:c['val_sse'])
    near=[c for c in good if c['val_sse']<=best['val_sse']+2*np.linalg.norm(c['val_prediction']-best['val_prediction'])/sigma+1e-10]
    return min(near,key=lambda c:(len(c['D']),c['parameters'],c['val_sse']))

def freeze():
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())['script_sha256']==HASH;return
    frozen=REPO/'src/train_dosy/_frozen'
    dump(p,dict(created_utc=datetime.now(timezone.utc).isoformat(),script_sha256=HASH,indices=INDICES,
        sources={str(p.relative_to(REPO)):sha(p) for p in frozen.rglob('*.py')},
        n_problems=36,methods=['RAI','DOME'],bounds=BOUNDS,alpha=.01,max_nfev=350,
        design='Full paired2x2: S=0 unrestricted amplitudes; S=1 mandatory conditional screening. P=0 preserve native-selected nominal order; P=1 select among all orders using reserved prediction rule. Every arm receives common refinement and final24row refit.',
        controls='Per problem and method, one common native proposal bank constructed from same18rows. All candidate supports/refits cached once; six internal validation rows, eight external rows.',
        reference='Additional fifth arm production_choice selects over unrestricted+screened candidates, reproducing paper S rule; it is not a factorial cell.',
        mask='Retain frozen benchmark mask (includes internal-validation information); matched posthoc mechanism analysis, not calibrated inference.',
        time_accounting='Measured common native proposal cost, screening cost, each candidate refinement, each final refit, and summed per-arm logical cost. Common bank is shared; logical totals are not independently measured end-to-end walltimes.',
        uncertainty='Paired problem bootstrap10000, seed2026101811, signal-only36problems. Difference-in-differences [(S1P1-S0P1)-(S1P0-S0P0)] is descriptive for this generator and native-order definition.',
        no_parameter_tuning=True,no_gpu=True))

def job(index):
    target=OUT/'results'/f'case_{index:03d}.json'
    if target.exists():return json.loads(target.read_text())
    z=generate(index,'test');mask=z['mask'];outer=z['train'];b=z['b'][outer];y=z['Y'][outer][:,mask];sigma=z['sigma']
    val=np.zeros(len(b),bool);val[2:-1:4]=True;tr=~val;groups=groups_from_ppm(z['ppm'][mask])
    inp=OUT/'inputs'/f'case_{index:03d}.npz';inp.parent.mkdir(exist_ok=True);np.savez_compressed(inp,**z)
    outputs=[];banks=[];start=time.perf_counter()
    for method in ['RAI','DOME']:
        methodstart=time.perf_counter()
        try:
            native_t=time.perf_counter();path,native=native_path(b[tr],y[tr],sigma,BOUNDS,method,4);native_seconds=time.perf_counter()-native_t
            candidates=[];screen_seconds=0.
            for bank_id,d in enumerate(path):
                raw=np.ones((len(d),y.shape[1]),bool);ss=time.perf_counter()
                screened=screen_support(b[tr],y[tr],sigma,d,groups,.01)[0] if len(d) else raw.copy()
                screening_seconds=time.perf_counter()-ss;screen_seconds+=screening_seconds
                for screen,sup in [(0,raw),(1,screened)]:
                    fit_t=time.perf_counter();c=refine(b[tr],y[tr],sigma,BOUNDS,d,sup,350);fit_seconds=time.perf_counter()-fit_t
                    pred=np.exp(-b[val,None]*c['D'])@c['A']
                    c.update(bank_id=bank_id,screen=screen,initial_D=d,initial_support=sup,refine_seconds=fit_seconds,screen_seconds=screening_seconds if screen else 0.,
                        val_prediction=pred,val_sse=float(np.sum(((pred-y[val])/sigma)**2)),training_sse=float(np.sum(((c['prediction']-y[tr])/sigma)**2)))
                    candidates.append(c)
            nominal=len(native['D']);raw=[c for c in candidates if not c['screen']]
            native_candidates=[c for c in raw if len(c['initial_D'])==nominal]
            native_chosen=min(native_candidates,key=lambda c:c['training_sse'])
            arms=[]
            for screen in [0,1]:
                bank=[c for c in candidates if c['screen']==screen]
                fixed=next(c for c in bank if c['bank_id']==native_chosen['bank_id'])
                arms.append((f'S{screen}P0',fixed,screen,0))
                arms.append((f'S{screen}P1',choose(bank,sigma),screen,1))
            arms.append(('production_choice',choose(candidates,sigma),None,None))
            for arm,c,screen,predict in arms:
                refit_t=time.perf_counter();f=refine(b,y,sigma,BOUNDS,c['D'],c['support'],350);refit_seconds=time.perf_counter()-refit_t
                r=metrics(z,f)
                used=candidates if screen is None else ([cc for cc in candidates if cc['screen']==screen] if predict else [c])
                logical=native_seconds+sum(cc['refine_seconds']+cc['screen_seconds'] for cc in used)+refit_seconds
                r.update(method=method,arm=arm,screen=screen,predict=predict,native_nominal_rank=nominal,chosen_bank_id=c['bank_id'],chosen_initial_rank=len(c['initial_D']),
                    chosen_screen=c['screen'],chosen_val_sse=c['val_sse'],chosen_numerical_success=c['success'],refit_seconds=refit_seconds,logical_seconds=logical,error=None)
                outputs.append(r)
                array=OUT/'arrays'/f'case_{index:03d}__{method}__{arm}.npz';array.parent.mkdir(exist_ok=True)
                np.savez_compressed(array,D=f['D'],A=f['A'],support=f['support'],prediction=f['prediction'])
            banks.append(dict(method=method,native_seconds=native_seconds,screen_seconds=screen_seconds,total_seconds=time.perf_counter()-methodstart,native_nominal_rank=nominal,
                candidates=[{k:c.get(k) for k in ['bank_id','screen','initial_D','D','parameters','success','kkt','stationarity','nfev','refine_seconds','screen_seconds','val_sse','training_sse']} for c in candidates]))
        except Exception as e:outputs.append(dict(method=method,error=type(e).__name__+': '+str(e),success=False))
    result=dict(index=index,id=z['id'],input_sha256=sha(inp),seconds=time.perf_counter()-start,outputs=outputs,banks=banks)
    dump(target,result);return result

def aggregate():
    cases=[json.loads(p.read_text()) for p in sorted((OUT/'results').glob('*.json'))];rows=[dict(index=c['index'],**o) for c in cases for o in c['outputs']]
    dump(OUT/'metrics.json',rows);summaries=[];contrasts=[];rng=np.random.default_rng(2026101811)
    keys=['spectral_leakage_fraction','unmatched_mass_fraction','test_clean_rmse_sigma','w1_logD','logical_seconds']
    for method in ['RAI','DOME']:
        rr=[r for r in rows if r['method']==method and r.get('error') is None]
        for arm in ['S0P0','S1P0','S0P1','S1P1','production_choice']:
            r=[q for q in rr if q['arm']==arm]
            if not r:continue
            summaries.append(dict(method=method,arm=arm,n=len(r),failures=sum(not q['success'] for q in r),candidate_failures=sum(not q['chosen_numerical_success'] for q in r),
                correct_order=sum(q['correct_rank'] for q in r),order_and_rates=sum(q['correct_rank'] and q['matched_components']==q['r_true'] for q in r),
                **{k:float(np.mean([q[k] for q in r])) for k in keys}))
        lookup={(r['index'],r['arm']):r for r in rr}
        for metric in keys[:-1]:
            vec=[]
            for i in INDICES:
                names=['S0P0','S1P0','S0P1','S1P1']
                if all((i,k) in lookup for k in names):vec.append([lookup[i,k][metric] for k in names])
            if not vec:continue
            vv=np.asarray(vec)
            for name,delta in [('screen_at_P0',vv[:,1]-vv[:,0]),('screen_at_P1',vv[:,3]-vv[:,2]),('prediction_at_S0',vv[:,2]-vv[:,0]),('prediction_at_S1',vv[:,3]-vv[:,1]),('interaction',vv[:,3]-vv[:,2]-vv[:,1]+vv[:,0])]:
                boot=np.mean(delta[rng.integers(0,len(delta),size=(10000,len(delta)))],axis=1)
                contrasts.append(dict(method=method,metric=metric,contrast=name,n=len(delta),mean=float(delta.mean()),paired_bootstrap95=np.quantile(boot,[.025,.975]).tolist()))
    result=dict(cases=len(cases),summaries=summaries,contrasts=contrasts,errors=[r for r in rows if r.get('error')],total_worker_seconds=sum(c['seconds'] for c in cases))
    dump(OUT/'summary.json',result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--aggregate',action='store_true');a=ap.parse_args();freeze()
    if not a.aggregate:
        start=time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
            fs=[pool.submit(job,i) for i in INDICES]
            for k,f in enumerate(concurrent.futures.as_completed(fs),1):f.result();print('factorial',k,'/36',round(time.perf_counter()-start,1),flush=True)
        dump(OUT/'run_timing.json',dict(seconds=time.perf_counter()-start,workers=a.workers))
    print(json.dumps(clean_json(aggregate()),indent=2))

if __name__=='__main__':main()
