"""Bounded CPU-only revision experiments; all frozen numerical code is imported unchanged.

Run from any directory with the repository's Python environment. Inputs,
per-case outcomes, configuration and source hashes are retained beside this
file. No test truth enters candidate construction or model selection.
"""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse, concurrent.futures, csv, hashlib, json, sys, time
from datetime import datetime, timezone
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.stats import beta

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent/'run_02'
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'benchmarks'))
from train_dosy._frozen.support import Config, fit_support, native_path, refine, screen_support, groups_from_ppm
from train_dosy.inversion import clean_json
from synthetic import generate
from evaluate import metrics

SOURCE_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
BOUNDS=(.1e-9,15e-9)
METHODS=('RAI','DOME')
NULL_COUNT=300
NULL_SEED_BASE=202610170000
ABLATION_INDICES=[i for i in range(32) if i%8<6]

def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean_json(data),indent=2,allow_nan=False),encoding='utf-8')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def null_job(index):
    file=OUT/'null_results'/f'null_{index:03d}.json'
    if file.exists():return json.loads(file.read_text())
    begin=time.perf_counter();seed=NULL_SEED_BASE+index;rng=np.random.default_rng(seed)
    sigma=(.02,.01,.005)[index%3];ppm=np.linspace(10,0,1024)
    b=np.linspace(0,1,32)**rng.uniform(1.5,2.4)*rng.uniform(1.3,2.7)*1e9
    Y=sigma*rng.normal(size=(32,1024));outer=np.arange(32)%4!=3
    inner_val=np.zeros(24,bool);inner_val[2:-1:4]=True
    discovery_rows=np.flatnonzero(outer)[~inner_val][:6]
    fixed=(ppm>4)&(ppm<4.65)
    discovery=binary_dilation(Y[discovery_rows].mean(0)>4*sigma/np.sqrt(len(discovery_rows)),iterations=2)
    input_path=OUT/'inputs'/f'null_{index:03d}.npz';input_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(input_path,Y=Y,b=b,ppm=ppm,sigma=sigma,outer_training_rows=outer,
                        internal_validation_positions=inner_val,discovery_rows=discovery_rows,
                        fixed_mask=fixed,discovery_mask=discovery,seed=seed)
    outcomes=[]
    for mode,mask in [('fixed_window',fixed),('independent_discovery',discovery)]:
        for method in METHODS:
            start=time.perf_counter()
            if not mask.any():
                q=dict(mode=mode,method=method,selected_columns=0,rank=0,
                       computation_status='no_selected_frequencies',fit_performed=False,
                       numerical_success=None,error=None,seconds=0.)
            else:
                try:
                    f=fit_support(b[outer],Y[outer][:,mask],sigma,ppm[mask],method=method)
                    q=dict(mode=mode,method=method,selected_columns=int(mask.sum()),rank=len(f['D']),
                           computation_status='fit_returned',fit_performed=True,numerical_success=f['success'],
                           error=None,seconds=time.perf_counter()-start,D=f['D'],A=f['A'],
                           kkt=f['kkt'],stationarity=f.get('stationarity'),candidate_kind=f['candidate_kind'],
                           candidate_diagnostics=f['candidate_diagnostics'])
                except Exception as e:
                    q=dict(mode=mode,method=method,selected_columns=int(mask.sum()),rank=None,
                           computation_status='error',fit_performed=True,numerical_success=False,
                           error=type(e).__name__+': '+str(e),seconds=time.perf_counter()-start)
            outcomes.append(q)
    result=dict(index=index,seed=seed,sigma=sigma,input_file=str(input_path.relative_to(OUT)),
                input_sha256=digest(input_path),seconds=time.perf_counter()-begin,outcomes=outcomes)
    dump(file,result);return result

def choose(candidates,sigma):
    good=[c for c in candidates if c['success'] and c['kkt']<=1e-7]
    if not good:raise RuntimeError('No numerically eligible candidate')
    best=min(good,key=lambda c:c['val_sse'])
    eligible=[c for c in good if c['val_sse']<=best['val_sse']+
              2*np.linalg.norm(c['val_prediction']-best['val_prediction'])/sigma+1e-10]
    return min(eligible,key=lambda c:(len(c['D']),c['parameters'],c['val_sse']))

def ablation_job(index):
    file=OUT/'ablation_results'/f'test_{index:03d}.json'
    if file.exists():return json.loads(file.read_text())
    begin=time.perf_counter();z=generate(index,'test');mask=z['mask'];outer=z['train']
    b=z['b'][outer];Y=z['Y'][outer][:,mask];sigma=z['sigma'];ppm=z['ppm'][mask]
    val=np.zeros(len(b),bool);val[2:-1:4]=True;tr=~val;groups=groups_from_ppm(ppm)
    input_path=OUT/'inputs'/f'ablation_{index:03d}.npz'
    np.savez_compressed(input_path,**z)
    outputs=[];method_audits=[]
    for method in METHODS:
        start=time.perf_counter()
        try:
            path,native=native_path(b[tr],Y[tr],sigma,BOUNDS,method,4)
            candidates=[]
            for d in path:
                raw=np.ones((len(d),Y.shape[1]),bool)
                supports=[('unrestricted',raw)]
                if len(d):
                    sup,_,_=screen_support(b[tr],Y[tr],sigma,d,groups,.01)
                    if not np.array_equal(sup,raw):supports.append(('screened',sup))
                for kind,sup in supports:
                    if any(np.array_equal(c['initial_D'],d) and np.array_equal(c['initial_support'],sup) for c in candidates):continue
                    c=refine(b[tr],Y[tr],sigma,BOUNDS,d,sup,350)
                    pred=np.exp(-b[val,None]*c['D'])@c['A']
                    c.update(kind=kind,initial_D=d,initial_support=sup,val_prediction=pred,
                             val_sse=float(np.sum(((pred-Y[val])/sigma)**2)),
                             training_sse=float(np.sum(((c['prediction']-Y[tr])/sigma)**2)))
                    candidates.append(c)
            raw=[c for c in candidates if c['kind']=='unrestricted']
            nominal=len(native['D'])
            native_order=[c for c in raw if len(c['initial_D'])==nominal]
            # Preserve the native order decision; use the same refined bank
            # and all-row refit as the other arms. This is NOT a rerun of the
            # original native-24-row estimator and not a complete 2x2 design.
            base=min(native_order,key=lambda c:c['training_sse'])
            selections=[('native_order_common_refit',base),('prediction_only',choose(raw,sigma)),
                        ('screen_and_prediction',choose(candidates,sigma))]
            method_audits.append(dict(method=method,path_seconds=time.perf_counter()-start,
                                      native_order=nominal,native_diagnostics=native.get('diagnostics'),
                                      candidates=[{k:c.get(k) for k in ('D','kind','parameters','success','kkt','stationarity','nfev','val_sse','training_sse')} for c in candidates]))
            for arm,c in selections:
                refit_start=time.perf_counter()
                f=refine(b,Y,sigma,BOUNDS,c['D'],c['support'],350)
                result=metrics(z,f)
                result.update(method=method,arm=arm,chosen_kind=c['kind'],
                              chosen_validation_sse=c['val_sse'],refit_seconds=time.perf_counter()-refit_start,error=None)
                outputs.append(result)
                arrays=OUT/'ablation_arrays'/f'{z["id"]}__{method}__{arm}.npz'
                arrays.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(arrays,D=f['D'],A=f['A'],support=f['support'],prediction=f['prediction'])
        except Exception as e:
            outputs.append(dict(method=method,error=type(e).__name__+': '+str(e)))
    result=dict(index=index,id=z['id'],input_file=str(input_path.relative_to(OUT)),
                input_sha256=digest(input_path),seconds=time.perf_counter()-begin,
                outcomes=outputs,method_audits=method_audits)
    dump(file,result);return result

def binomial(k,n):
    if not n:return None
    return dict(events=k,n=n,rate=k/n,
                exact_two_sided_95=[0. if k==0 else beta.ppf(.025,k,n-k+1),
                                    1. if k==n else beta.ppf(.975,k+1,n-k)],
                exact_one_sided_upper_95=1. if k==n else beta.ppf(.95,k+1,n-k))

def aggregate():
    nulls=[json.loads(p.read_text()) for p in sorted((OUT/'null_results').glob('*.json'))]
    null_summary=[]
    for mode in ['fixed_window','independent_discovery']:
        for method in METHODS:
            rows=[o for n in nulls for o in n['outcomes'] if o['mode']==mode and o['method']==method]
            valid=[o for o in rows if o['error'] is None];performed=[o for o in valid if o['fit_performed']]
            null_summary.append(dict(mode=mode,method=method,errors=len(rows)-len(valid),
                nonzero_outputs=binomial(sum(o['rank']>0 for o in valid),len(valid)),
                nonzero_among_fitted=binomial(sum(o['rank']>0 for o in performed),len(performed)),
                no_selected_frequencies=sum(not o['fit_performed'] for o in valid),
                numerical_failures=sum(o['numerical_success'] is False for o in valid),
                total_fit_seconds=sum(o['seconds'] for o in rows)))
    ablations=[json.loads(p.read_text()) for p in sorted((OUT/'ablation_results').glob('*.json'))]
    summary=[]
    for method in METHODS:
        for arm in ['native_order_common_refit','prediction_only','screen_and_prediction']:
            rows=[o for a in ablations for o in a['outcomes'] if o['method']==method and o.get('arm')==arm]
            if not rows:continue
            summary.append(dict(method=method,arm=arm,n=len(rows),
                correct_order=sum(o['correct_rank'] for o in rows),
                order_and_rates=sum(o['correct_rank'] and o['matched_components']==o['r_true'] for o in rows),
                numerical_failures=sum(not o['success'] for o in rows),
                **{k:float(np.mean([o[k] for o in rows])) for k in ['spectral_leakage_fraction','unmatched_mass_fraction','test_clean_rmse_sigma','w1_logD']}))
    errors=[o for a in ablations for o in a['outcomes'] if o['error'] is not None]
    result=dict(source_sha256=SOURCE_SHA,null_cases=len(nulls),null_summary=null_summary,
                ablation_cases=len(ablations),ablation_summary=summary,ablation_errors=errors,
                total_worker_seconds=sum(n['seconds'] for n in nulls)+sum(a['seconds'] for a in ablations),
                scope='Synthetic supplement of frozen atomic RAI-S/DOME-S; not TRAIn-MF signed-v2.2; no GPU/training; no experimental validation.')
    dump(OUT/'summary.json',result)
    rows=[{**o,'case':a['id']} for a in ablations for o in a['outcomes']]
    if rows:
        with (OUT/'ablation_metrics.csv').open('w',newline='',encoding='utf-8') as f:
            fields=sorted(set().union(*(r.keys() for r in rows)));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--mode',choices=['all','null','ablation','aggregate'],default='all');a=p.parse_args()
    protocol=OUT/'protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text())['source_sha256']==SOURCE_SHA,'Changed script: use a new output directory/version.'
    else:
        frozen=ROOT/'src/train_dosy/_frozen'
        dump(protocol,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=SOURCE_SHA,
             frozen_source_sha256={str(p.relative_to(ROOT)):digest(p) for p in frozen.rglob('*.py')},
             null_count=NULL_COUNT,null_seed_base=NULL_SEED_BASE,null_seeds=list(range(NULL_SEED_BASE,NULL_SEED_BASE+NULL_COUNT)),
             null_sigma_cycle=[.02,.01,.005],null_fixed_window_ppm=[4,4.65],
             null_discovery='First six low-b outer-training rows after excluding internal validation; mean >4sigma/sqrt(6), two dilation positions; empty mask stops without fit.',
             null_outcome='Any nonzero returned rank, regardless of final numerical-success flag; errors listed separately and excluded from rate denominator.',
             ablation_indices=ABLATION_INDICES,ablation_scope='24 previously frozen signal-present test problems; paired post-hoc mechanism study, no tuning.',
             ablation_arms=['native_order_common_refit','prediction_only','screen_and_prediction'],
             matched_controls='One common native proposal bank per method on18rows, same simplex-free NNLS variable-projection refinement,max_nfev350; six validation rows for predictive arms; final24row refit for all.',
             limitations='Native-order arm retains native order but uses common refinement; no screening-only arm, no full factorial attribution, no equal wall-clock-budget claim. Original benchmark masks retained for ablation and include internal validation information.',
             config=vars(Config()),workers=a.workers,blas_threads_per_worker=1))
    (OUT/'inputs').mkdir(exist_ok=True)
    start=time.perf_counter()
    for label,function,indices in [('null',null_job,range(NULL_COUNT)),('ablation',ablation_job,ABLATION_INDICES)]:
        if a.mode not in ('all',label):continue
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(function,i) for i in indices]
            for done,f in enumerate(concurrent.futures.as_completed(futures),1):
                f.result()
                if done%25==0 or done==len(indices):print(label,done,'/',len(indices),'wall_s',round(time.perf_counter()-start,1),flush=True)
    result=aggregate();print(json.dumps(clean_json(result),indent=2),flush=True)

if __name__=='__main__':main()
