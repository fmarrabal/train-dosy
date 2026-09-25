import numpy as np
from scipy.ndimage import binary_dilation
from .models import FitRequest

def clean_json(value):
    if isinstance(value,np.ndarray): return clean_json(value.tolist())
    if isinstance(value,np.generic): return clean_json(value.item())
    if isinstance(value,dict): return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean_json(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value): return None
    return value

def prepare(request):
    q=request if isinstance(request,FitRequest) else FitRequest.model_validate(request)
    y=np.asarray(q.Y);b=np.asarray(q.b);ppm=np.asarray(q.ppm)
    # Discovery excludes internal validation rows, including for MF. Caller masks
    # must likewise be predeclared or constructed without validation/test data.
    order=np.argsort(b); val=np.zeros(len(b),bool);val[2:-1:4]=True
    discovery=order[~val][:6]
    mask=np.asarray(q.mask,bool) if q.mask is not None else binary_dilation(
        y[discovery].mean(0)>4*q.sigma/np.sqrt(len(discovery)),iterations=2)
    if not mask.any(): raise ValueError("No signal frequencies detected; check phase, sigma and orientation")
    return q,y,b,ppm,mask

def fit(request):
    """No chemical truth or requested rank is accepted by this interface."""
    q,y,b,ppm,mask=prepare(request)
    if q.method=="MF-AUTO":
        from .mf import fit_mf
        return fit_mf(q,y,b,ppm,mask)
    from ._frozen.support import fit_support, Config
    out=fit_support(b,y[:,mask],q.sigma,ppm[mask],q.diffusion_bounds,
        method=q.method.split('-')[0],config=Config(max_components=q.max_components))
    # X stores bin mass, not density; off-grid rates remain authoritative.
    edges=np.linspace(*np.log(q.diffusion_bounds),q.bins+1)
    x=np.zeros((q.bins,int(mask.sum())))
    if len(out['D']):
        indices=np.clip(np.searchsorted(edges,np.log(out['D']),side='right')-1,0,q.bins-1)
        np.add.at(x,indices,out['A'])
    order=len(out['D'])
    return clean_json(dict(schema_version="1.0",software_version="0.1.0",method=q.method,
        D=out['D'],A=out['A'],X=x,logD_edges=edges,prediction=out['prediction'],
        ppm=ppm[mask],selected_frequency_indices=np.flatnonzero(mask),mask=mask,
        spectral_support=out['support'],selected_rank=order,
        selection_mode="automatic",candidate_ranks=sorted({len(c['D']) for c in out['candidate_diagnostics']}),
        search_limit=q.max_components,search_limit_reached=order==q.max_components,
        success=out['success'],kkt=out['kkt'],stationarity=out.get('stationarity',0.),
        diagnostics={k:out[k] for k in ['candidate_diagnostics','inner_validation_rows',
            'candidate_kind','atomic_model_mismatch_flag','bound_hit','poor_rate_precision',
            'rate_information_condition','local_log_rate_sd','warning']},
        units=dict(b="s/m^2",D="m^2/s",X="signal mass per log-D bin"),
        excluded_frequencies="not estimated; zero intensity is not inferred"))
