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
            D=np.geomspace(*q.diffusion_bounds,q.bins)))
        quote=lambda p:str(p).replace("'","''")
        command=f"addpath('{quote(directory)}'); run_mf_api('{quote(source)}','{quote(target)}');"
        proc=subprocess.run([executable,'-batch',command],capture_output=True,text=True,timeout=600)
        if proc.returncode or not target.is_file():
            raise RuntimeError('MATLAB MF computation failed; inspect configuration and toolbox availability')
        out=json.loads(target.read_text(encoding='utf-8'))
    # MATLAB JSON collapses singleton dimensions; restore the public schema.
    r=int(out['selected_rank']) if out.get('selected_rank') is not None else None
    if r is None or not out.get('X'):
        raise RuntimeError('MF automatic selection unresolved; no usable stationary candidate')
    p=int(mask.sum())
    out['X']=np.asarray(out['X']).reshape(q.bins,p)
    if r:out['A']=np.asarray(out['A']).reshape(r,p)
    else:out['A']=np.zeros((0,p))
    out['D_grid']=np.asarray(out['D_grid']).ravel()
    out['prediction']=np.exp(-b[:,None]*out['D_grid'])@out['X']
    out.update(schema_version='1.0',software_version='0.1.0',method='MF-AUTO',
        ppm=ppm[mask],mask=mask,selected_frequency_indices=np.flatnonzero(mask),
        selection_mode='automatic',search_limit=4,search_limit_reached=r==4,
        units=dict(b='s/m^2',D_grid='m^2/s',X='signal mass per diffusion node'),
        excluded_frequencies='not estimated; zero intensity is not inferred')
    return clean_json(out)
