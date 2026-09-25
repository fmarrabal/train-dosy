from pathlib import Path
import json, hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
BENCH=ROOT.parent/'ilt_unified_benchmark_20260924'

def serial(x):
    if isinstance(x,np.ndarray):return serial(x.tolist())
    if isinstance(x,np.generic):return serial(x.item())
    if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    if isinstance(x,Path):return str(x)
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def dump(path,value):
    Path(path).write_text(json.dumps(serial(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def archived(case):
    from scipy.io import loadmat
    z=loadmat(BENCH/'inputs'/f'{case}.mat',simplify_cells=True)
    mask=loadmat(BENCH/'results'/f'{case}_train.mat',variable_names=['mask'],simplify_cells=True)['mask'].astype(bool)
    return dict(id=case,b=z['x']*1e9,Y=z['Y'],Y_clean=z['Y_clean'],sigma=float(z['noise_sigma']),
        ppm=z['ppm'],mask=mask,train=z['train'].astype(bool),test=z['test'].astype(bool),
        truth_D=np.atleast_1d(z['truth_centers'])*1e-9,truth_A=np.atleast_2d(z['truth_A']),truth_widths=np.atleast_1d(z['truth_widths']),
        scenario=str(z['scenario']),r_true=int(z['r_true']),snr=float(z['snr']))
