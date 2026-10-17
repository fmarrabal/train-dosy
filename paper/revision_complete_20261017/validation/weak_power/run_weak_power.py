from pathlib import Path
import os,sys,time,json,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment
from scipy.stats import beta
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'benchmarks')]
from train_dosy._frozen.support import fit_support
from train_dosy.inversion import clean_json
from synthetic import generate
from evaluate import metrics

def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean_json(x),indent=2,allow_nan=False),encoding='utf-8')
def problem(i):
    index=800003+8*i;z=generate(index,'test');snr=(50,100,200)[(i//8)%3];level=(.02,.04,.08,.16)[i//24]
    a=z['truth_A'].copy();a[-1]*=level/.04;a/=a.sum(0).max();z['truth_A']=a;z['snr']=snr;z['sigma']=1/snr
    clean=np.exp(-z['b'][:,None]*z['truth_D'])@a;z['Y_clean']=clean
    rng=np.random.default_rng(202610181000+i);z['Y']=clean+rng.normal(0,z['sigma'],clean.shape)
    tr=z['train'];val=np.zeros(tr.sum(),bool);val[2:-1:4]=True
    mask_rows=np.flatnonzero(tr)[~val][:6]
    z['mask']=binary_dilation(z['Y'][mask_rows].mean(0)>4*z['sigma']/np.sqrt(6),iterations=2)
    z['level']=level;z['revision_seed']=202610181000+i
    return z
def job(i):
    file=OUT/'results'/f'{i:03d}.json'
    if file.exists():return json.loads(file.read_text())
    z=problem(i);mask=z['mask'];tr=z['train'];out=[]
    for method in ('RAI','DOME'):
        t=time.perf_counter()
        try:
            f=fit_support(z['b'][tr],z['Y'][tr][:,mask],z['sigma'],z['ppm'][mask],method=method)
            m=metrics(z,f)
            out.append(dict(method=method,success=f['success'],metrics=m,error=None,seconds=time.perf_counter()-t,D=f['D']))
        except Exception as ex:out.append(dict(method=method,success=False,metrics={},error=type(ex).__name__+': '+str(ex),seconds=time.perf_counter()-t))
    selected_minority_mass=float(z['truth_A'][-1,mask].sum()/z['truth_A'][-1].sum())
    record=dict(index=i,snr=z['snr'],level=z['level'],seed=z['revision_seed'],selected_minority_mass=selected_minority_mass,selected_channels=int(mask.sum()),outcomes=out)
    inp=OUT/'inputs'/f'{i:03d}.npz';inp.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(inp,b=z['b'],Y=z['Y'],Y_clean=z['Y_clean'],truth_D=z['truth_D'],truth_A=z['truth_A'],ppm=z['ppm'],mask=mask,train=tr)
    record['input_sha256']=hashlib.sha256(inp.read_bytes()).hexdigest();dump(file,record);return record
def summary(rows):
    table=[]
    for level in (.02,.04,.08,.16):
        for snr in (50,100,200):
            for method in ('RAI','DOME'):
                records=[r for r in rows if r['level']==level and r['snr']==snr]
                hits=[o for r in records for o in r['outcomes'] if o['method']==method]
                success=sum(o['metrics'].get('minority_rate_recovered',False) for o in hits);n=len(hits)
                lo=0 if success==0 else float(beta.ppf(.025,success,n-success+1));hi=1 if success==n else float(beta.ppf(.975,success+1,n-success))
                table.append(dict(method=method,minority_amplitude_factor=level,snr=snr,n=n,recovered=success,interval95=[lo,hi],numerical_success=sum(o['success'] for o in hits),mean_mask_minority_recall=float(np.mean([r['selected_minority_mass'] for r in records]))))
    dump(OUT/'summary.json',dict(rows=table,n_problems=len(rows),scope='Factor-level recovery counts include failures; pairs are not independent method replications. No calibrated species probability.'))
if __name__=='__main__':
    dump(OUT/'protocol.json',dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),n=96,seeds=list(range(202610181000,202610181096)),levels=[.02,.04,.08,.16],snr=[50,100,200],replicates_per_cell=8,mask='training-only first six low-b candidate-fitting acquisitions; excludes inner validation and outer test',methods=['RAI-S','DOME-S'],frozen=True))
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(job,i) for i in range(96)]
        for j,f in enumerate(as_completed(fs),1):
            rows.append(f.result())
            if j%12==0:print('Finished',j,'of96',flush=True)
    summary(rows)
