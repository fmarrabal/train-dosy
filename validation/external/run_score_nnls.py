"""Prespecified positive-amplitude SCORE sensitivity; primary sources unchanged."""
import run_external_benchmark as base
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np,time
from dome.solver import amplitudes_exact

original_fit=base.fit_method
def with_nnls(name,b,y,sigma,ppm):
    if name!='SCORE_NNLS':return original_fit(name,b,y,sigma,ppm)
    oldcfg=base.loc.Config;oldamp=base.loc.amplitudes
    def config(**kw):kw['nonnegative']=True;return oldcfg(**kw)
    def amplitudes(k,y,nonnegative=False):return amplitudes_exact(k,y) if nonnegative else oldamp(k,y,False)
    base.loc.Config=config;base.loc.amplitudes=amplitudes
    try:
        q=base.score_fit(b,y,sigma)
        return q['D'],q['amplitudes'],q['success'],dict(evaluations=q['evaluations'],candidates=q['selection_candidates'],rank=len(q['D']),variant='Positive-amplitude sensitivity of SCORE with unchanged CV wrapper; not signed GNAT default')
    finally:base.loc.Config=oldcfg;base.loc.amplitudes=oldamp
base.fit_method=with_nnls

if __name__=='__main__':
    config=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_sha256=base.sha(Path(__file__)),primary_script_sha256=base.sha(Path(base.__file__)),method='SCORE_NNLS',selection='Same as SCORE primary, including null candidate and physical-rate eligibility',budgets='Same 5000 Nelder-Mead evaluations',scope='Additional amplitude-constraint sensitivity, distinct from signed published default; no modification of primary protocol.')
    base.dump(base.OUT/'score_nnls_protocol.json',config)
    jobs=[(i,'iid','SCORE_NNLS') for i in base.IDS]+[(i,k,'SCORE_NNLS') for k in ('baseline','correlated') for i in base.STRESS_IDS]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for j,f in enumerate(as_completed([pool.submit(base.job,*job) for job in jobs]),1):
            r=f.result()
            if j%12==0:print('SCORE-NNLS',j,'of',len(jobs),flush=True)
