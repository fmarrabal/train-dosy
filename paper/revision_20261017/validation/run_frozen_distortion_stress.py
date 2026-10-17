"""Prespecified numerical stress; perturbations are not instrument-calibrated."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,concurrent.futures,csv,hashlib,json,sys,time
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent/'distortion_stress'
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'benchmarks'))
from train_dosy._frozen.support import fit_support,native_path
from train_dosy.inversion import clean_json
from synthetic import generate
from evaluate import metrics
SOURCE_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
INDICES=list(range(6))+list(range(8,14));KINDS=['spectral_ar1','constant_baseline','impulsive_contamination','b_calibration']
METHODS=['RAI','RAI_S','DOME','DOME_S']

def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean_json(data),indent=2,allow_nan=False),encoding='utf-8')

def job(index,kind):
    target=OUT/'results'/f'{index:03d}_{kind}.json'
    if target.exists():return json.loads(target.read_text())
    z=generate(index,'test');sig=z['sigma'];seed=202610180000+10*index+KINDS.index(kind);rng=np.random.default_rng(seed)
    background=z['Y']-z['Y_clean'];params={}
    if kind=='spectral_ar1':
        rho=.6;v=rng.normal(size=z['Y'].shape)
        for j in range(1,v.shape[1]):v[:,j]=rho*v[:,j-1]+np.sqrt(1-rho*rho)*v[:,j]
        z['Y']=z['Y_clean']+sig*v;params={'rho':rho,'marginal_sigma':sig}
    elif kind=='constant_baseline':
        z['Y']=z['Y']+sig;params={'offset_in_sigma':1.,'offset':sig,'attenuation':'none'}
    elif kind=='impulsive_contamination':
        hits=rng.random(z['Y'].shape)<.005
        signs=rng.choice([-1.,1.],size=z['Y'].shape)
        z['Y']=z['Y']+hits*signs*8*sig
        params={'probability':.005,'impulse_sigma':8.,'contaminated_cells_total':int(hits.sum()),
                'sigma_contract':'Supplied sigma remains the uncontaminated Gaussian background SD.'}
    elif kind=='b_calibration':
        z['Y_clean']=np.exp(-1.05*z['b'][:,None]*z['truth_D'])@z['truth_A']
        z['Y']=z['Y_clean']+background
        params={'physical_b_over_supplied_b':1.05,'physical_rates_unchanged':True}
    path=OUT/'inputs'/f'{index:03d}_{kind}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**z,distortion_seed=seed)
    rows=[];m=z['mask'];tr=z['train'];begin=time.perf_counter()
    for method in METHODS:
        start=time.perf_counter()
        try:
            if method.endswith('_S'):
                f=fit_support(z['b'][tr],z['Y'][tr][:,m],sig,z['ppm'][m],method=method[:-2])
                numerical=bool(f['success']);diagnostics={k:f[k] for k in ['kkt','stationarity','candidate_kind','bound_hit','atomic_model_mismatch_flag'] if k in f}
            else:
                _,f=native_path(z['b'][tr],z['Y'][tr][:,m],sig,(.1e-9,15e-9),method,4)
                diagnostics=f.get('diagnostics',{})
                numerical=diagnostics.get('optimizer_success')
                f['success']=numerical if numerical is not None else True
            row=metrics(z,f);row.update(method=method,kind=kind,seconds=time.perf_counter()-start,
                    error=None,numerical_status_reported=numerical,diagnostics=diagnostics)
            arrays=OUT/'arrays'/f'{index:03d}_{kind}_{method}.npz';arrays.parent.mkdir(exist_ok=True)
            np.savez_compressed(arrays,D=f['D'],A=f['A'],mask=m)
            rows.append(row)
        except Exception as e:rows.append(dict(method=method,kind=kind,error=type(e).__name__+': '+str(e),seconds=time.perf_counter()-start))
    out=dict(id=z['id'],index=index,kind=kind,distortion_seed=seed,parameters=params,
             input_file=str(path.relative_to(OUT)),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             seconds=time.perf_counter()-begin,results=rows)
    dump(target,out);return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    protocol=OUT/'protocol.json'
    if protocol.exists():assert json.loads(protocol.read_text())['source_sha256']==SOURCE_SHA
    else:
        frozen=ROOT/'src/train_dosy/_frozen'
        dump(protocol,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=SOURCE_SHA,
             frozen_source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.rglob('*.py')},
             indices=INDICES,kinds=KINDS,methods=METHODS,seeds={f'{i}_{k}':202610180000+10*i+KINDS.index(k) for i in INDICES for k in KINDS},
             parameters={'spectral_ar1_rho':.6,'nonattenuated_baseline_sigma':1.,'impulse_probability':.005,'impulse_sigma':8.,'physical_b_over_supplied_b':1.05},
             masks='Original frozen outer-training signal masks kept fixed; no mask retuning under distortion.',
             partition='32 acquisitions;24outer-training/8external;nativeuses24;Suses18/6internal then24refit, asoriginalpaper.',
             truth_access='Only input generation and final metrics; never fit/selection.',
             scope='Experiment-inspired synthetic stress; parameters chosen before outcomes, not fitted to an instrument; no experimental validation; no corrections to frozen solvers.',
             sigma='Original known iid Gaussian background SD; correct marginal SD for AR1; does not absorb baseline, impulse contamination or b calibration.',workers=a.workers,blas_threads_per_worker=1))
    start=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(job,i,k) for i in INDICES for k in KINDS]
        for n,f in enumerate(concurrent.futures.as_completed(futures),1):
            f.result()
            if n%12==0:print('stress',n,'/48 wall_s',round(time.perf_counter()-start,1),flush=True)
    cases=[json.loads(p.read_text()) for p in sorted((OUT/'results').glob('*.json'))]
    rows=[r for c in cases for r in c['results']];summaries=[]
    for kind in KINDS:
        for method in METHODS:
            q=[r for r in rows if r['kind']==kind and r['method']==method];valid=[r for r in q if r['error'] is None]
            summaries.append(dict(kind=kind,method=method,n=len(valid),errors=len(q)-len(valid),
                correct_order=sum(r['correct_rank'] for r in valid),
                order_and_rates=sum(r['correct_rank'] and r['matched_components']==r['r_true'] for r in valid),
                numerical_failures=sum(r['numerical_status_reported'] is False for r in valid),
                **{k:float(np.mean([r[k] for r in valid])) if valid else None for k in ['spectral_leakage_fraction','unmatched_mass_fraction','test_clean_rmse_sigma','w1_logD']}))
    result=dict(source_sha256=SOURCE_SHA,n_cases=len(cases),n_fits=len(rows),summary=summaries,
                errors=[r for r in rows if r['error'] is not None],total_worker_seconds=sum(c['seconds'] for c in cases),
                wall_seconds=time.perf_counter()-start,scope='Synthetic stress only; not calibrated experimental error levels.')
    dump(OUT/'summary.json',result)
    flat=[{k:v for k,v in r.items() if k!='diagnostics'} for r in rows]
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        fields=sorted(set().union(*(r.keys() for r in flat)));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(flat)
    print(json.dumps(clean_json(result),indent=2))

if __name__=='__main__':main()
