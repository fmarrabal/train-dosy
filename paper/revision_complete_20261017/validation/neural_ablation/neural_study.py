"""Prespecified CPU-only multi-seed RAI-Net study; archived solvers are read-only."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,gzip,hashlib,time,argparse,concurrent.futures,platform,shutil
from datetime import datetime,timezone
from dataclasses import asdict
import numpy as np
from scipy.ndimage import binary_dilation

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[3]
ARCHIVE=Path('E:/ARTICULOS-CIENTIFICOS/20240905_OLIGOQUITOSANO/CODIGO_MEJORADO/dosy_support_v2_20260925')
sys.path.insert(0,str(ARCHIVE/'vendor'));sys.path.insert(0,str(ARCHIVE))
from rai_ilt.adaptive import solve_adaptive,AdaptiveConfig
from rai_ilt.model import DecayProblem,bin_kernel
from rai_ilt.net import train_policy,RAINetPolicy
from evaluate import map_metrics
from synthetic import generate

TRAIN_SEEDS=[172901,172902,172903,172904,172905]
TEST_SEED_BASE=202610180000
SCENARIOS=('single','double','triple','weak','close','proportional','null','broad')
CFG=AdaptiveConfig(max_rounds=6,policy_shortlist=1)
SOURCE_HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v

def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(clean(v),indent=2,allow_nan=False),encoding='utf-8')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class ClassicalShortlist:
    """Original deterministic proposal priority with identical neural shortlist logic."""
    def rank(self,features):return np.arange(len(features),dtype=int)

def freeze():
    path=OUT/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())['script_sha256']==SOURCE_HASH,'New script requires new study directory'
        return
    traces=sorted((ARCHIVE/'training').glob('trace_*.json.gz'))
    assert len(traces)==80
    dump(path,dict(created_utc=datetime.now(timezone.utc).isoformat(),script_sha256=SOURCE_HASH,
        original_sources={str(p.relative_to(ARCHIVE)):sha(p) for p in [ARCHIVE/'vendor/rai_ilt/adaptive.py',ARCHIVE/'vendor/rai_ilt/net.py',ARCHIVE/'vendor/rai_ilt/tree.py',ARCHIVE/'vendor/rai_ilt/model.py',ARCHIVE/'evaluate.py',ARCHIVE/'synthetic.py']},
        training_trace_sha256={p.name:sha(p) for p in traces},training_problem_seed_range=[610000,610079],
        training_seeds=TRAIN_SEEDS,training=dict(epochs=400,hidden=48,learning_rate=.003,split='64 whole training problems and16 whole validation problems, resampled by each declared seed; independent neural initializations, shared trace bank'),
        test_count=120,test_seeds=list(range(TEST_SEED_BASE,TEST_SEED_BASE+120)),
        domains={'ID':list(range(72)),'OOD_noise':list(range(72,96)),'OOD_acquisition':list(range(96,120))},
        generator='8 families crossed with SNR50/100/200 (ID72), SNR25/35/500 (OODnoise24), and SNR50/100/200 with b-exponent0.7..1.1 and maxb0.7..1.2e9 (OODacquisition24). All new whole problems.',
        mask='Mean first six fitting rows excluding internal and external validation, >4sigma/sqrt6, dilate2; fixed4<ppm<4.65 null window; no truth selection.',
        config=asdict(CFG),methods=['classical_shortlist']+[f'net_seed_{s}' for s in TRAIN_SEEDS],
        budget='Identical six-round cap, one-candidate shortlist with same numerical fallback, three penalty paths and 3000 inner iterations. Measured actual fits/seconds reported; this matches caps and decision machinery, not wallclock or realized fit counts.',
        primary_endpoints=['signal-present mass-weighted logD W1','external clean RMSE/sigma'],
        uncertainty='Paired problem bootstrap10000 with seed2026101899; seed-averaged neural-minus-classical per problem, stratified by domain; per-training-seed results and seed range separate, not600 independent cases.',
        no_test_tuning=True,gpu_used=False,python=sys.version,platform=platform.platform()))

def train():
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    traces=[]
    for p in sorted((ARCHIVE/'training').glob('trace_*.json.gz')):
        with gzip.open(p,'rt',encoding='utf-8') as f:traces.append(json.load(f))
    records=[r for t in traces for r in t['records'] if r.get('validation_gain') is not None]
    assert len({r['problem_id'] for r in records})==80
    for seed in TRAIN_SEEDS:
        cp=OUT/'checkpoints'/f'policy_{seed}.npz';report=OUT/'training'/f'seed_{seed}.json'
        if report.exists():continue
        start=time.perf_counter();policy,history=train_policy(records,seed=seed,epochs=400,hidden=48,learning_rate=.003)
        policy.save(cp)
        tr=set(policy.metadata['train_problem_ids']);va=set(policy.metadata['validation_problem_ids'])
        assert not tr&va and len(tr)==64 and len(va)==16
        dump(report,dict(seed=seed,seconds=time.perf_counter()-start,checkpoint_sha256=sha(cp),metadata=policy.metadata,
            best_epoch=min(history,key=lambda h:h['validation_loss'])['epoch'],history=history,
            test_data_used=False,truth_used=False,torch_version=torch.__version__,device='cpu'))
        print('trained',seed,'seconds',round(time.perf_counter()-start,2),flush=True)

def new_problem(index):
    seed=TEST_SEED_BASE+index;rng=np.random.default_rng(seed);scenario=SCENARIOS[index%8]
    domain='ID' if index<72 else ('OOD_noise' if index<96 else 'OOD_acquisition')
    snr=(25,35,500)[(index-72)//8] if domain=='OOD_noise' else (50,100,200)[(index//8)%3]
    r={'single':1,'double':2,'null':0}.get(scenario,3)
    ppm=np.linspace(10,0,1024)
    b=np.linspace(0,1,32)**rng.uniform(*((.7,1.1) if domain=='OOD_acquisition' else (1.5,2.4)))*rng.uniform(*((.7,1.2) if domain=='OOD_acquisition' else (1.3,2.7)))*1e9
    centers=np.exp(np.log([.65,2.3,7.1])[:r]+rng.normal(0,.10,r))
    if scenario=='close':centers=np.array([.8,1.05,5.5])*np.exp(rng.normal(0,.02,3))
    a=np.zeros((r,len(ppm)));peaks=np.array([7.73,5.35,4.20,3.2,2.05,1.35])+rng.normal(0,.04,6)
    for j in range(r):
        for center,amp in [(peaks[0]+.018*j,rng.uniform(.35,1.)),(peaks[1+j],rng.uniform(.5,1.)),(peaks[4+j%2],rng.uniform(.3,.8))]:
            a[j]+=amp*np.exp(-.5*((ppm-center)/rng.uniform(.03,.06))**2)
    if scenario=='proportional':a=np.array([1.,.7,.45])[:,None]*a[0]
    if scenario=='weak':a[-1]*=.04
    if r:a/=a.sum(0).max()
    widths=np.full(r,.18 if scenario=='broad' else 0.)
    if scenario=='broad':
        nodes,weights=np.polynomial.hermite.hermgauss(24)
        curves=np.column_stack([np.exp(-b[:,None]*d*1e-9*np.exp(np.sqrt(2)*w*nodes))@(weights/np.sqrt(np.pi)) for d,w in zip(centers,widths)])
    else:curves=np.exp(-b[:,None]*centers*1e-9)
    clean_y=curves@a;sigma=1/snr;y=clean_y+rng.normal(0,sigma,clean_y.shape)
    test=np.arange(32)%4==3;tr=~test;val=np.zeros(24,bool);val[2:-1:4]=True
    discover=np.flatnonzero(tr)[~val][:6]
    mask=binary_dilation(y[discover].mean(0)>4*sigma/np.sqrt(6),iterations=2)
    if scenario=='null':mask=(ppm>4)&(ppm<4.65)
    return dict(id=f'independent_{index:03d}_{domain}_{scenario}',index=index,seed=seed,scenario=scenario,domain=domain,r_true=r,snr=snr,
        b=b,Y=y,Y_clean=clean_y,sigma=sigma,ppm=ppm,mask=mask,train=tr,test=test,discovery_rows=discover,
        truth_D=centers*1e-9,truth_A=a,truth_widths=widths)

def solve_one(z,method):
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    policy=ClassicalShortlist() if method=='classical_shortlist' else RAINetPolicy.load(OUT/'checkpoints'/f'policy_{method.split("_")[-1]}.npz')
    start=time.perf_counter();mask=z['mask'];tr=z['train']
    q=solve_adaptive(DecayProblem(z['b'][tr],z['Y'][tr][:,mask],z['sigma'],(.1e-9,15e-9),z['ppm'][mask]),CFG,policy=policy)
    seconds=time.perf_counter()-start;pred=bin_kernel(z['b'],q.rate_edges)@q.masses
    report=map_metrics(z,q.masses,pred)
    if not z['r_true']:report['w1_logD']=None
    report.update(method=method,domain=z.get('domain','pilot'),seconds=seconds,index=z.get('index'),
        success=q.diagnostics['optimizer']['success'],relative_gap=q.diagnostics['optimizer']['relative_gap'],
        fixed_tree_fits=q.diagnostics['fixed_tree_fits'],policy_fallbacks=q.diagnostics['policy_fallbacks'],
        selected_penalty=q.diagnostics['selected_penalty'],active_bins=q.diagnostics['active_bins'],mass=float(q.masses.sum()),
        observation_sha256=hashlib.sha256(z['Y'][tr][:,mask].tobytes()).hexdigest(),error=None)
    return report,q,pred

def pilot():
    z=generate(0,'net_validation');z['domain']='timing_pilot'
    for method in ['classical_shortlist',f'net_seed_{TRAIN_SEEDS[0]}']:
        r,q,_=solve_one(z,method);dump(OUT/'pilot'/f'{method}.json',dict(metrics=r,diagnostics=q.diagnostics))
        print('pilot',method,json.dumps(r),flush=True)

def job(index):
    z=new_problem(index);p=OUT/'inputs'/f'case_{index:03d}.npz';p.parent.mkdir(exist_ok=True)
    if not p.exists():np.savez_compressed(p,**z)
    reports=[]
    methods=['classical_shortlist']+[f'net_seed_{s}' for s in TRAIN_SEEDS]
    # Rotate ordering without changing pairings; no outcome-adaptive scheduling.
    methods=methods[index%len(methods):]+methods[:index%len(methods)]
    for method in methods:
        result=OUT/'results'/f'case_{index:03d}__{method}.json'
        if result.exists():reports.append(json.loads(result.read_text())['metrics']);continue
        try:
            r,q,pred=solve_one(z,method)
            arr=OUT/'arrays'/f'case_{index:03d}__{method}.npz';arr.parent.mkdir(exist_ok=True)
            np.savez_compressed(arr,X=q.masses,edges=q.rate_edges,prediction=pred)
            payload=dict(metrics=r,diagnostics=q.diagnostics,history=q.history,array_sha256=sha(arr),input_sha256=sha(p))
        except Exception as e:
            r=dict(index=index,id=z['id'],domain=z['domain'],scenario=z['scenario'],snr=z['snr'],r_true=z['r_true'],method=method,success=False,error=type(e).__name__+': '+str(e))
            payload=dict(metrics=r)
        dump(result,payload);reports.append(r)
    return reports

def aggregate():
    rows=[json.loads(p.read_text())['metrics'] for p in sorted((OUT/'results').glob('*.json'))]
    dump(OUT/'metrics.json',rows)
    summaries=[];deltas=[];rng=np.random.default_rng(2026101899)
    for domain in ['all','ID','OOD_noise','OOD_acquisition']:
        rr=[r for r in rows if domain=='all' or r['domain']==domain]
        for method in ['classical_shortlist']+[f'net_seed_{s}' for s in TRAIN_SEEDS]:
            x=[r for r in rr if r['method']==method];ok=[r for r in x if r.get('error') is None]
            summaries.append(dict(domain=domain,method=method,n=len(x),errors=len(x)-len(ok),failures=sum(not r['success'] for r in ok),
                **{k:float(np.mean([r[k] for r in ok if r.get(k) is not None])) if any(r.get(k) is not None for r in ok) else None for k in ['w1_logD','test_clean_rmse_sigma','seconds','fixed_tree_fits','mass']},
                n_signal=sum(r['r_true']>0 for r in ok),n_null=sum(r['r_true']==0 for r in ok)))
        indices=sorted({r['index'] for r in rr});lookup={(r['index'],r['method']):r for r in rr}
        for metric in ['w1_logD','test_clean_rmse_sigma','seconds','fixed_tree_fits']:
            pairs=[];byseed={s:[] for s in TRAIN_SEEDS}
            for i in indices:
                b=lookup.get((i,'classical_shortlist'));nn=[lookup.get((i,f'net_seed_{s}')) for s in TRAIN_SEEDS]
                if b is None or b.get('error') or b.get(metric) is None or any(n is None or n.get('error') or n.get(metric) is None for n in nn):continue
                dd=[n[metric]-b[metric] for n in nn];pairs.append(np.mean(dd))
                for s,d in zip(TRAIN_SEEDS,dd):byseed[s].append(d)
            a=np.asarray(pairs)
            if not len(a):continue
            boot=np.mean(a[rng.integers(0,len(a),size=(10000,len(a)))],axis=1)
            deltas.append(dict(domain=domain,metric=metric,n_problems=len(a),difference='average of five neural policies minus classical shortlist',
                mean=float(a.mean()),paired_problem_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),
                seed_mean_differences={s:float(np.mean(v)) for s,v in byseed.items()},
                inference_scope='Conditional on shared training trace bank and five declared training runs; problems are independent, seed outputs within a problem are paired, not independent replications.'))
    result=dict(rows=len(rows),summaries=summaries,paired_differences=deltas,errors=[r for r in rows if r.get('error')],
        numerical_failures=[{k:r[k] for k in ('index','method','relative_gap')} for r in rows if not r['success'] and not r.get('error')],
        total_fit_seconds=sum(r.get('seconds',0) for r in rows),all_expected_completed=len(rows)==720)
    dump(OUT/'summary.json',result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['train','pilot','run','aggregate']);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args()
    freeze()
    if a.mode=='train':train()
    elif a.mode=='pilot':pilot()
    elif a.mode=='run':
        start=time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(job,i) for i in range(120)]
            for k,f in enumerate(concurrent.futures.as_completed(futures),1):
                r=f.result();print('completed',k,'/120','case',r[0]['index'],'wall_s',round(time.perf_counter()-start,1),flush=True)
        dump(OUT/'run_timing.json',dict(seconds=time.perf_counter()-start,workers=a.workers));aggregate()
    else:print(json.dumps(clean(aggregate()),indent=2))

if __name__=='__main__':main()
