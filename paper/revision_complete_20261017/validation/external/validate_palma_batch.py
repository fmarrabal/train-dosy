"""Archive fixed-iteration arithmetic validation; not scientific validation."""
from pathlib import Path
import os,sys,json,hashlib
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
sys.path.insert(0,str(REPO.parent/'diffusion-inversion/python/src'))
from diffusion_inversion.entropy import palma_fit
from palma_batch import fit

rng=np.random.default_rng(20261017008)
b=np.linspace(.02,2.0,24)
d=np.geomspace(.1,15,256)
k=np.exp(-b[:,None]*d)
scales=np.array([.2,1.,7.])
truth=np.zeros((256,3));truth[[55,130,201],np.arange(3)]=scales
y=k@truth+rng.normal(0,.004,(24,3))*scales
radius=.003
rows=[]
for weight in (.01,.1,.5):
    x,diag=fit(k,y/scales,radius/scales,weight,200,1e-12)
    original,cost,ok,it,native=palma_fit(k,y,scales,radius,weight,200,1e-12)
    physical=x*scales
    error=np.linalg.norm(k@(physical-original))/max(1,np.linalg.norm(k@original))
    mass_error=np.max(np.abs(physical-original))
    assert not diag['converged'] and not ok
    assert error<1e-11 and mass_error<1e-10,(weight,error,mass_error)
    rows.append(dict(weight=weight,prediction_relative_error=float(error),
        maximum_mass_difference=float(mass_error),batch_converged=diag['converged'],
        native_converged=ok,iterations_per_column=200,scales=scales.tolist()))
report=dict(scope='Fixed-iteration PPXA+ arithmetic equivalence only. All cases unresolved. The normalized batch certificate and physical-scale native stopping criteria need not stop at the same iteration.',
    cases=rows,pass_=True,
    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'palma_batch.py',REPO.parent/'diffusion-inversion/python/src/diffusion_inversion/entropy.py']})
(HERE/'palma_batch_arithmetic_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
