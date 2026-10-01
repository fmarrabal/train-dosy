"""Optional local licensed MATLAB backend; no solver reimplementation."""
from pathlib import Path
import json, os, subprocess, tempfile
import numpy as np
from scipy.io import savemat
from .inversion import clean_json

def fit_mf(q,y,b,ppm,mask):
    executable=os.environ.get('TRAIN_DOSY_MATLAB')
    directory=Path(os.environ.get('TRAIN_DOSY_MATLAB_FUNCTIONS',Path(__file__).resolve().parents[2]/'matlab'))
    if not executable or not (directory/'run_mf_api.m').is_file():
        raise RuntimeError('MF backend unavailable: configure TRAIN_DOSY_MATLAB and TRAIN_DOSY_MATLAB_FUNCTIONS')
    with tempfile.TemporaryDirectory(prefix='train_dosy_') as temp:
        temp=Path(temp); source=temp/'input.mat';target=temp/'output.json'
        savemat(source,dict(Y=y[:,mask],b=b,ppm=ppm[mask],sigma=q.sigma,
            D=np.geomspace(*q.diffusion_bounds,q.bins),method=q.method,max_components=q.max_components))
        quote=lambda p:str(p).replace("'","''")
        command=f"addpath('{quote(directory)}'); run_mf_api('{quote(source)}','{quote(target)}');"
        proc=subprocess.run([executable,'-batch',command],capture_output=True,text=True,timeout=600)
        if proc.returncode or not target.is_file():
            raise RuntimeError('MATLAB MF computation failed; inspect configuration and toolbox availability')
        out=json.loads(target.read_text(encoding='utf-8'))
    # MATLAB JSON collapses singleton dimensions; restore the public schema.
    r=int(out['selected_rank']) if out.get('selected_rank') is not None else None
    if r is None or r<0 or out.get('X') is None or np.asarray(out['X']).size==0:
        raise RuntimeError('MF automatic selection unresolved; no usable stationary candidate')
    active=int(out.get('active_rank',out.get('diagnostics',{}).get('active_rank',r)))
    if not 0<=active<=r:
        raise RuntimeError('MATLAB MF returned an inconsistent active factor count')
    p=int(mask.sum())
    out['X']=np.asarray(out['X']).reshape(q.bins,p)
    # MATLAB serializes both nD-by-0 and 0-by-p matrices as []. Restore both
    # sides of the null factorization; X=0 remains a usable rank-zero result.
    # signed-v2.2 also removes exactly inactive factors while retaining the
    # nominal selected rank for review. Frozen MF-AUTO keeps its prior rank.
    out['A']=np.asarray(out['A']).reshape(active,p)
    out['S']=np.asarray(out['S']).reshape(q.bins,active)
    out['D_grid']=np.asarray(out['D_grid']).ravel()
    if isinstance(out.get('kkt'),list) and not out['kkt']:
        out['kkt']=None  # MATLAB [] is not a nullable scalar for the C# SDK.
    out['prediction']=np.exp(-b[:,None]*out['D_grid'])@out['X']
    out.update(schema_version='1.0',software_version='0.2.1',method=q.method,
        ppm=ppm[mask],mask=mask,selected_frequency_indices=np.flatnonzero(mask),
        selection_mode='automatic',search_limit=q.max_components,search_limit_reached=r==q.max_components,
        units=dict(b='s/m^2',D_grid='m^2/s',X='signal mass per diffusion node'),
        excluded_frequencies='not estimated; zero intensity is not inferred')
    return clean_json(out)
