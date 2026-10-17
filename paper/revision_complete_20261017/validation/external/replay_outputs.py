"""Recompute saved forward predictions and errors without refitting models."""
from pathlib import Path
import json,hashlib
import numpy as np
import run_external_benchmark as source

HERE=Path(__file__).resolve().parent
files=sorted((HERE/'results').glob('*.json'))
assert len(files)==504
sources=json.loads((HERE/'protocol.json').read_text())['sources']
def resolve_recorded(name):
    normalized=name.replace('\\','/')
    for folder,root in [('train-dosy-public',source.ROOT),('diffusion-inversion',source.DI)]:
        marker='/'+folder+'/'
        if marker in normalized:return root/normalized.split(marker,1)[1]
    p=Path(name)
    if p.is_file():return p
    raise FileNotFoundError(name)
source_checks={p:hashlib.sha256(resolve_recorded(p).read_bytes()).hexdigest()==h for p,h in sources.items()}
assert all(source_checks.values()),source_checks
problems={};errors=[];max_forward=0.;max_metric=0.;outside={}
for file in files:
    r=json.loads(file.read_text());assert r['error'] is None,r
    key=(r['kind'],r['index'])
    if key not in problems:problems[key]=source.problem(r['index'],r['kind'])
    z=problems[key];a=np.load(HERE/'outputs'/file.with_suffix('.npz').name)
    prediction=source.kernel(z['b'],a['D'])@a['A']
    assert np.isfinite(prediction).all()
    delta=np.max(abs(prediction-a['prediction']),initial=0.)
    max_forward=max(max_forward,float(delta))
    assert delta<1e-11
    for target,name in [('Y_clean','test_clean_rmse_sigma'),('Y','test_observed_rmse_sigma')]:
        value=float(np.sqrt(np.mean((prediction[z['test']]-z[target][z['test']][:,z['mask']])**2))/z['sigma'])
        diff=abs(value-r['metrics'][name]);max_metric=max(max_metric,diff)
        assert diff<1e-10,(file,name,value,r['metrics'][name])
    if r['kind']=='iid' and r['index']!=0 and np.any((a['D']<source.BOUNDS[0])|(a['D']>source.BOUNDS[1])):
        outside.setdefault(r['method'],[]).append(r['index'])
report=dict(fits_verified=len(files),input_conditions=len(problems),maximum_forward_difference=max_forward,
    maximum_rmse_difference=max_metric,source_hashes_unchanged=source_checks,
    iid_outside_nominal_interval=outside,passed=True,
    scope='Saved-output arithmetic and source-integrity replay, not independent refitting or chemical validation.')
(HERE/'replay_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
