"""Prespecified external comparison; truth is used only by the evaluator."""
from pathlib import Path
import os,sys,json,time,hashlib,warnings,subprocess,re
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy.optimize import nnls
from scipy.stats import chi2
from scipy.ndimage import binary_dilation
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
DI=ROOT.parent/'diffusion-inversion'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'benchmarks'),str(DI/'python/src')]
from train_dosy._frozen.support import fit_support,groups_from_ppm
from train_dosy.inversion import clean_json
from synthetic import generate
from evaluate import map_metrics,metrics,grid
from diffusion_inversion import Problem,solve_grid
from diffusion_inversion import locodosy_native as loc

BOUNDS=(.1e-9,15e-9)
EDGES=np.linspace(np.log(BOUNDS[0]),np.log(BOUNDS[1]),257)
DGRID=np.exp((EDGES[:-1]+EDGES[1:])/2)
METHODS=['RAI_S','DOME_S','SCORE','LOCODOSY','Tikhonov','PALMA']
IDS=list(range(48))
STRESS_IDS=[i for i in range(16) if i%8<6]
def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean_json(obj),indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def kernel(b,d):return np.exp(-np.asarray(b)[:,None]*np.asarray(d)[None,:])

def windows(ppm):
    g=groups_from_ppm(ppm)
    return np.asarray([[np.flatnonzero(g==i).min(),np.flatnonzero(g==i).max()+1] for i in np.unique(g)],int)

def score_fit(b,y,sigma):
    """GNAT objective/default initialization port, common held-out order wrapper."""
    val=np.zeros(len(b),bool);val[2:-1:4]=True
    cfg=loc.Config(max_components=4,max_evaluations=5000,nonnegative=False)
    candidates=[]
    candidates.append(dict(D=np.zeros(0),amplitudes=np.zeros((0,y.shape[1])),success=True,evaluations=0))
    for r in range(1,5):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');q=loc.score(b[~val],y[~val],r,cfg)
        candidates.append(q)
    for q in candidates:
        pred=kernel(b[val],q['D'])@q['amplitudes']
        q['val_prediction']=pred;q['val_sse']=float(np.sum((pred-y[val])**2)/sigma**2)
        q['physical']=bool(np.all((q['D']>=BOUNDS[0])&(q['D']<=BOUNDS[1])) and np.isfinite(pred).all())
    good=[q for q in candidates if q['success'] and q['physical']]
    best=min(good,key=lambda q:q['val_sse'])
    eligible=[q for q in good if q['val_sse']<=best['val_sse']+2*np.linalg.norm(q['val_prediction']-best['val_prediction'])/sigma+1e-10]
    chosen=min(eligible,key=lambda q:(len(q['D']),q['val_sse']))
    r=len(chosen['D'])
    if r:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');f=loc.score(b,y,r,cfg)
    else:f=dict(D=np.zeros(0),amplitudes=np.zeros((0,y.shape[1])),success=True,evaluations=0)
    f['selection_candidates']=[dict(rank=len(q['D']),success=q['success'],physical=q['physical'],val_sse=q['val_sse']) for q in candidates]
    return f

def fit_method(name,b,y,sigma,ppm):
    if name in ('RAI_S','DOME_S'):
        q=fit_support(b,y,sigma,ppm,method=name[:-2]);return q['D'],q['A'],q['success'],dict(kkt=q['kkt'],stationarity=q.get('stationarity'),rank=len(q['D']))
    if name=='SCORE':
        q=score_fit(b,y,sigma);return q['D'],q['amplitudes'],q['success'],dict(evaluations=q['evaluations'],candidates=q['selection_candidates'],rank=len(q['D']))
    if name=='LOCODOSY':
        cfg=loc.Config(max_components=4,max_evaluations=5000,max_diffusion=BOUNDS[1],diffusion_bins=256,nonnegative=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');q=loc.fit(b,y,ppm,windows=windows(ppm),config=cfg)
        okay=not q['failures'] and all(w.get('success',False) for w in q['windows'])
        return q['D'],q['amplitudes'],okay,dict(local_orders=[len(w.get('D',[])) for w in q['windows']],failures=q['failures'],unstable_windows=q['unstable_windows'],rank_is_not_global_species=True)
    if name=='Tikhonov':
        val=np.zeros(len(b),bool);val[2:-1:4]=True
        candidates=[]
        for reg in (.01,1.,100.,10000.):
            q=solve_grid(Problem(b[~val],y[~val],sigma),DGRID,method='tikhonov',regularization=reg,order=2,tolerance=1e-7)
            candidates.append((np.sum((kernel(b[val],DGRID)@q.masses-y[val])**2),reg))
        _,reg=min(candidates)
        q=solve_grid(Problem(b,y,sigma),DGRID,method='tikhonov',regularization=reg,order=2,tolerance=1e-7)
        return DGRID,q.masses,q.converged,dict(regularization=reg,inner_validation_loss=[list(c) for c in candidates],kkt=q.diagnostics['kkt_inf'])
    if name=='PALMA':
        from palma_batch import fit as palma
        val=np.zeros(len(b),bool);val[2:-1:4]=True
        scales=np.maximum(y[~val][:4].mean(0),sigma)
        candidates=[]
        for weight in (.01,.1,.5):
            radius=float(sigma*np.sqrt(chi2.ppf(.95,(~val).sum())))
            x,diag=palma(kernel(b[~val],DGRID),y[~val]/scales,radius/scales,weight,1000,1e-5)
            candidates.append((float(np.sum((kernel(b[val],DGRID)@(x*scales)-y[val])**2)),weight))
        _,weight=min(candidates)
        radius=float(sigma*np.sqrt(chi2.ppf(.95,len(b))))
        x,diag=palma(kernel(b,DGRID),y/scales,radius/scales,weight,5000,1e-5)
        diag.update(scales=scales,inner_candidates=candidates,scope='PALMA objective with training-only low-b entropy scale and validation-selected weight; no upstream performance identity claim')
        return DGRID,x*scales,diag['converged'],diag
    raise ValueError(name)

def problem(index,kind):
    z=generate(index,'test')
    if kind!='iid':
        rng=np.random.default_rng(202610180000+index)
        if kind=='baseline':z['Y']=z['Y']+z['sigma']
        elif kind=='correlated':
            noise=rng.normal(size=z['Y'].shape)*z['sigma']
            for i in range(1,noise.shape[1]):noise[:,i]=.6*noise[:,i-1]+.8*noise[:,i]
            z['Y']=z['Y_clean']+noise
    return z

def job(index,kind,name):
    file=OUT/'results'/f'{kind}_{index:03d}_{name}.json'
    if file.exists():return json.loads(file.read_text())
    z=problem(index,kind);mask=z['mask'];tr=z['train'];started=time.perf_counter()
    try:
        d,a,okay,diag=fit_method(name,z['b'][tr],z['Y'][tr][:,mask],z['sigma'],z['ppm'][mask])
        pred=kernel(z['b'],d)@a
        physical=bool(np.all((d>=BOUNDS[0])&(d<=BOUNDS[1])) and np.all(a>=-1e-10) and np.isfinite(pred).all())
        if physical:
            if name in ('PALMA','Tikhonov'):x=a
            else:_,x=grid(d,np.maximum(a,0))
            metric=map_metrics(z,x,pred)
            if name in ('RAI_S','DOME_S','SCORE'):metric.update(metrics(z,dict(D=d,A=np.maximum(a,0),success=okay)))
        else:metric={}
        te=z['test'];metric['test_clean_rmse_sigma']=float(np.sqrt(np.mean((pred[te]-z['Y_clean'][te][:,mask])**2))/z['sigma'])
        metric['test_observed_rmse_sigma']=float(np.sqrt(np.mean((pred[te]-z['Y'][te][:,mask])**2))/z['sigma'])
        result=dict(index=index,kind=kind,method=name,scenario=z['scenario'],snr=z['snr'],selected_channels=int(mask.sum()),success=bool(okay),physical=physical,metrics=metric,diagnostics=diag,error=None,seconds=time.perf_counter()-started)
        arrays=OUT/'outputs'/f'{kind}_{index:03d}_{name}.npz';arrays.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(arrays,D=d,A=a,prediction=pred)
    except Exception as exc:
        result=dict(index=index,kind=kind,method=name,scenario=z['scenario'],snr=z['snr'],success=False,physical=False,metrics={},error=type(exc).__name__+': '+str(exc),seconds=time.perf_counter()-started)
    dump(file,result);return result

def protocol():
    sources=[Path(__file__),ROOT/'src/train_dosy/_frozen/support.py',DI/'python/src/diffusion_inversion/locodosy_native.py',DI/'python/src/diffusion_inversion/solvers.py',DI/'python/src/diffusion_inversion/entropy.py']
    sources.append(OUT/'palma_batch.py')
    return dict(frozen_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),sources={str(p):sha(p) for p in sources},methods=METHODS,ids=IDS,stress_ids=STRESS_IDS,kinds=['iid','baseline','correlated'],bins=256,bounds=BOUNDS,tikhonov_lambdas=[.01,1,100,10000],palma_weights=[.01,.1,.5],palma_entropy_scale='max(mean first four internal-fitting low-b rows,sigma), retained in refit',palma_budget=5000,palma_selection_budget=1000,palma_tol=1e-5,score_budget=5000,scope='Shared synthetic matrices, unchanged signal masks, no experimental or universal superiority claim; signed amplitudes preclude positive-measure metrics when negative.',external_repo=str(DI))

def summarize():
    rows=[json.loads(p.read_text()) for p in (OUT/'results').glob('*.json')]
    result=[]
    for kind in ('iid','baseline','correlated'):
        for name in METHODS:
            group=[r for r in rows if r['kind']==kind and r['method']==name]
            if not group:continue
            item=dict(kind=kind,method=name,n=len(group),numerical_success=sum(r['success'] for r in group),physical=sum(r['physical'] for r in group),errors=sum(r['error'] is not None for r in group),seconds=sum(r['seconds'] for r in group))
            for key in ('test_clean_rmse_sigma','test_observed_rmse_sigma','w1_logD','spectral_leakage_fraction','unmatched_mass_fraction'):
                vals=[r['metrics'][key] for r in group if key in r['metrics'] and r['scenario']!='null']
                item[key]=float(np.mean(vals)) if vals else None
                item[key+'_n']=len(vals)
            result.append(item)
    dump(OUT/'summary.json',dict(rows=result,total_fits=len(rows)))
    print(json.dumps(clean_json(result),indent=2))

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--workers',type=int,default=2);ap.add_argument('--summarize',action='store_true');args=ap.parse_args()
    if args.summarize:summarize();raise SystemExit
    protocol_file=OUT/'protocol.json'
    if not protocol_file.exists():dump(protocol_file,protocol())
    if args.pilot:
        for name in METHODS:
            r=job(0,'iid',name);print(name,r['seconds'],r['success'],r['error'],flush=True)
    else:
        jobs=[(i,'iid',name) for i in IDS for name in METHODS]+[(i,k,name) for k in ('baseline','correlated') for i in STRESS_IDS for name in METHODS]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(job,*j) for j in jobs]
            for count,f in enumerate(as_completed(futures),1):
                r=f.result()
                if count%12==0:print('Completed',count,'of',len(jobs),r['method'],r['seconds'],flush=True)
    summarize()
