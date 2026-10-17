"""Python translation/adaptation of GNAT LOCODOSY, SCORE, OUTSCORE and DECRA.

Original MATLAB copyright Mathias Nilsson (2007, 2010, 2017), GPL-2.0-or-later
where stated; GNAT distribution license GPLv3. Translation dated 2026-09-09,
GPL-3.0-or-later. Changed language, indexing, errors and headless interfaces.
This is existing LOCODOSY methodology, not an original inversion algorithm.
"""
from dataclasses import dataclass,asdict
import time
import numpy as np
from scipy.optimize import minimize,least_squares,nnls
from scipy.special import erf


@dataclass(frozen=True)
class Config:
    method: str = 'score'
    max_components: int = 3
    svd_cutoff: float = .99
    auto_reduce: bool = True
    min_percent: float = 5.
    max_diffusion: float = 5e-9
    diffusion_bins: int = 256
    threshold_percent: float = 5.
    padding_points: int = 1
    max_evaluations: int = 10000
    nonnegative: bool = False

    def __post_init__(self):
        if self.method not in ('score','outscore','decra'):raise ValueError('method must be score, outscore or decra')
        for key in ('max_components','diffusion_bins','padding_points','max_evaluations'):
            if type(getattr(self,key)) is not int:raise ValueError(key+' must be an integer')
        if not 1<=self.max_components<=8:raise ValueError('max_components must be between 1 and 8')
        if self.diffusion_bins<2 or self.padding_points<0 or self.max_evaluations<1:raise ValueError('Invalid grid, padding or evaluation budget')
        if not np.isfinite(self.svd_cutoff) or not 0<self.svd_cutoff<1:raise ValueError('svd_cutoff must be in (0,1)')
        if not np.isfinite(self.min_percent) or not 0<=self.min_percent<100:raise ValueError('min_percent must be in [0,100)')
        if not np.isfinite(self.max_diffusion) or self.max_diffusion<=0:raise ValueError('max_diffusion must be positive')
        if not np.isfinite(self.threshold_percent) or not 0<self.threshold_percent<100:raise ValueError('threshold_percent must be in (0,100)')
        if type(self.nonnegative) is not bool or type(self.auto_reduce) is not bool:raise ValueError('Boolean options required')
        if self.method=='decra' and self.nonnegative:raise ValueError('Native DECRA has no NNLS amplitude constraint; use SCORE for nonnegative fits')


def observations(b,y):
    if np.iscomplexobj(b) or np.iscomplexobj(y):raise ValueError('Real, phased spectra and real b are required')
    b=np.asarray(b,float);y=np.asarray(y,float)
    if b.ndim!=1:raise ValueError('b must be one-dimensional')
    if y.ndim==1:y=y[:,None]
    if y.ndim!=2 or y.shape[0]!=len(b) or not y.shape[1]:raise ValueError('Y must be attenuations x frequencies')
    if len(b)<4 or not np.isfinite(b).all() or not np.isfinite(y).all():raise ValueError('At least four finite attenuation rows are required')
    if np.any(b<0) or np.any(np.diff(b)<=0):raise ValueError('b must be nonnegative and strictly increasing; sort acquisition rows first')
    return b,y


def segment(first_spectrum,threshold_percent=5.,padding_points=1):
    """Threshold crossings plus bounded padding and merging (GUI adaptation).

    Padding is explicit in points; GNAT GUI uses spectrometer MHz to express
    25 Hz. Edge clipping and merging overlapping intervals are added here.
    Returns disjoint [start,stop) intervals in zero-based Python indexing.
    """
    s=np.asarray(first_spectrum,float)
    if s.ndim!=1 or not s.size or not np.isfinite(s).all():raise ValueError('Finite spectrum vector required')
    if not 0<threshold_percent<100 or type(padding_points) is not int or padding_points<0:raise ValueError('Invalid segmentation options')
    if np.max(s)<=0:return np.empty((0,2),int)
    active=s>np.max(s)*threshold_percent/100
    changes=np.diff(np.r_[False,active,False].astype(int))
    starts=np.flatnonzero(changes==1);stops=np.flatnonzero(changes==-1)
    windows=[]
    for a,z in zip(starts,stops):
        a=max(0,int(a)-padding_points);z=min(len(s),int(z)+padding_points)
        if windows and a<=windows[-1][1]:windows[-1][1]=max(z,windows[-1][1])
        else:windows.append([a,z])
    return np.asarray(windows,int).reshape(-1,2)


def estimate_order(y,cutoff=.99,maximum=3):
    """GNAT cumulative singular-value SUM, deliberately not squared energy."""
    if not 0<cutoff<1 or maximum<1:raise ValueError('Invalid SVD cutoff or maximum')
    singular=np.linalg.svd(y,compute_uv=False)
    if singular.sum()==0:return 1
    return int(min(maximum,np.searchsorted(np.cumsum(singular)/singular.sum(),cutoff,side='right')+1,len(singular)))


def kernel(b,d):
    with np.errstate(over='ignore',invalid='ignore'):
        k=np.exp(-b[:,None]*np.asarray(d)[None,:])
    if not np.isfinite(k).all():raise FloatingPointError('Nonfinite exponential kernel')
    return k


def amplitudes(k,y,nonnegative=False):
    if nonnegative:return np.column_stack([nnls(k,y[:,j],maxiter=max(100,20*k.shape[1]))[0] for j in range(y.shape[1])])
    return np.linalg.lstsq(k,y,rcond=None)[0]


def score_objective(d10,b,y,method='score',nonnegative=False):
    try:
        k=kernel(b,np.asarray(d10)*1e-10);a=amplitudes(k,y,nonnegative)
        if method=='outscore' and len(d10)>1:
            total=np.abs(a).sum(axis=1)
            if np.any(total==0):return 1e100
            norm=np.abs(a)/total[:,None]
            return float(y.shape[1]+sum(np.dot(norm[i],norm[j]) for i in range(len(a)) for j in range(i+1,len(a))))
        return float(np.sum((k@a-y)**2))
    except (ValueError,FloatingPointError,np.linalg.LinAlgError):return 1e100


def score(b,y,q,cfg):
    # Same dscale=1e-10, summed-signal initialization and Nelder-Mead simplex
    # as score_mn. No bound or prior D is silently added to the objective.
    total=y.sum(axis=1)
    def mono(x):
        try:return float(np.sum((x[0]*kernel(b,[x[1]*1e-10])[:,0]-total)**2))
        except FloatingPointError:return 1e100
    initial=minimize(mono,[total[0],4.],method='Nelder-Mead',
                     options=dict(xatol=1e-4,fatol=1e-4,maxiter=400,maxfev=cfg.max_evaluations))
    center=initial.x[1]
    factor=(q//2)*1.5
    guess=np.array([center]) if q==1 else np.linspace(center/factor,center*factor,q)
    opt=minimize(score_objective,guess,args=(b,y,cfg.method,cfg.nonnegative),method='Nelder-Mead',
                 options=dict(xatol=1e-6,fatol=1e-6,maxiter=10000,maxfev=cfg.max_evaluations))
    d=np.sort(opt.x)*1e-10;k=kernel(b,d);a=amplitudes(k,y,cfg.nonnegative)
    return dict(D=d,amplitudes=a,relative_contribution=np.abs(a).sum(axis=1),
                success=bool(opt.success),message=str(opt.message),evaluations=int(opt.nfev),objective=float(opt.fun),
                initialization_success=bool(initial.success),initial_D=guess*1e-10)


def decra(b,y,q):
    """Algebraic translation of decra_mn, preserving native signed amplitudes."""
    step=np.diff(b)
    if np.max(abs(step-step.mean()))>1e-8*np.max(step):raise ValueError('DECRA requires uniformly spaced b')
    if q>min(y.shape[1],len(b)-1):raise ValueError('DECRA order exceeds matrix dimensions')
    vc,s,ut=np.linalg.svd(y[:-1].T,full_matrices=False)
    sc=np.diag(s[:q]);uc=ut[:q].T;vc=vc[:,:q]
    if s[q-1]<=np.finfo(float).eps*s[0]*max(y.shape):raise ValueError('DECRA requested order has numerically singular SVD')
    ev,v=np.linalg.eig((uc.T@y[1:]@vc)@np.linalg.inv(sc))
    if np.max(abs(ev.imag))>1e-12 or np.any(ev.real<=0):raise ValueError('DECRA produced nonpositive/complex eigenvalues')
    order=np.argsort(ev.real);ev=ev.real[order];v=np.real(v[:,order])
    pspec=np.linalg.pinv(vc@np.linalg.inv(sc)@v);pint=uc@v
    scale=np.linalg.lstsq(pint,y[:-1].sum(axis=1),rcond=None)[0]
    if np.any(scale==0):raise ValueError('DECRA zero component scaling')
    pint=pint*scale[None,:];pspec=pspec/scale[:,None]
    pint2=pint*ev[None,:]
    combined=np.vstack([pint[0],(pint[1:]+pint2[:-1])/2,pint2[-1]])
    d=-np.log(ev)/step[0];k=kernel(b,d)
    a0=np.sum(k*combined,axis=0)/np.sum(k*k,axis=0)
    a=a0[:,None]*pspec;order=np.argsort(d)
    return dict(D=d[order],amplitudes=a[order],relative_contribution=a0[order],success=True,
                message='Native DECRA algebra, without NNLS refit',evaluations=0,objective=float(np.sum((k@a-y)**2)))


def exclusion_reasons(result,cfg):
    d=result['D'];rel=np.asarray(result['relative_contribution']);total=rel.sum()
    percent=100*rel/total if total!=0 else np.full(len(rel),np.nan)
    reasons=[]
    if np.any(d>cfg.max_diffusion):reasons.append('D above limit')
    if np.any(d<0):reasons.append('negative D')
    if np.any(percent>100):reasons.append('relative contribution above 100%')
    if np.any(percent<cfg.min_percent):reasons.append('relative contribution below min_percent')
    # MATLAB rounds half away from zero; scale is D / 1e-10 then 2 decimals.
    rounded=np.sign(d)*np.floor(abs(d)*1e12+.5)
    if len(np.unique(rounded))<len(d):reasons.append('duplicate D at native rounding precision')
    if not np.isfinite(percent).all():reasons.append('undefined relative contribution')
    return reasons


def legacy_error_fit(b,y,d,a):
    """Native fixed-spectra scale/D refit and n_b-2q standard-error convention.

    These legacy display widths are not calibrated confidence intervals.
    Ill-conditioned Jacobians are flagged and assigned undefined (NaN) errors.
    """
    q=len(d);guess=np.linspace(.95,1.05,2*q);guess[1::2]*=d/1e-10
    def fun(x):return (kernel(b,x[1::2]*1e-10)@(x[::2,None]*a)-y).ravel(order='F')
    def jac(x):
        k=kernel(b,x[1::2]*1e-10);j=np.empty((y.size,2*q))
        for z in range(q):
            j[:,2*z]=(k[:,[z]]*a[[z],:]).ravel(order='F')
            j[:,2*z+1]=(-b[:,None]*1e-10*k[:,[z]]*x[2*z]*a[[z],:]).ravel(order='F')
        return j
    try:
        opt=least_squares(fun,guess,jac=jac,ftol=1e-8,xtol=1e-8,gtol=1e-8,max_nfev=200*(2*q))
        rss=float(np.sum(opt.fun**2));degrees=len(b)-2*q;j=jac(opt.x);cond=float(np.linalg.cond(j))
        valid=degrees>0 and np.isfinite(cond) and cond<1e12
        cov=np.linalg.inv(j.T@j) if valid else None
        if valid and np.any(np.diag(cov)<0):valid=False
        errors=np.sqrt(rss/degrees)*np.sqrt(np.diag(cov)) if valid else np.full(2*q,np.nan)
        return dict(scales=opt.x[::2],D=opt.x[1::2]*1e-10,standard_error_D=errors[1::2]*1e-10,
                    success=bool(opt.success),uncertainty_defined=bool(valid),jacobian_condition=cond,rss=rss,
                    degrees_of_freedom=degrees,interpretation='GNAT legacy display error; not a calibrated interval')
    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
        return dict(scales=np.ones(q),D=d.copy(),standard_error_D=np.full(q,np.nan),success=False,
                    uncertainty_defined=False,error=str(exc),interpretation='Undefined legacy error fit')


def fit_window(b,y,q,config=None):
    cfg=config or Config();b,y=observations(b,y)
    if type(q) is not int or not 1<=q<=min(cfg.max_components,y.shape[1],len(b)-1):raise ValueError('Invalid local component count')
    history=[]
    while True:
        try:
            r=decra(b,y,q) if cfg.method=='decra' else score(b,y,q,cfg)
            reasons=exclusion_reasons(r,cfg)
            history.append(dict(q=q,exclusion_reasons=reasons,success=r['success']))
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            history.append(dict(q=q,error=str(exc)));reasons=['numerically invalid model'];r=None
        reduce=cfg.auto_reduce and cfg.method!='outscore' and bool(reasons) and q>1
        if reduce:q-=1;continue
        if r is None:raise ValueError('Local fit failed: '+str(history))
        break
    r.update(selected_order=q,reduction_history=history,exclusion_reasons=reasons,
             prediction=kernel(b,r['D'])@r['amplitudes'])
    condition=float(np.linalg.cond(kernel(b,r['D'])))
    expansion=float(np.max(abs(r['amplitudes']))/max(np.max(abs(y)),1e-300))
    r['numerical_diagnostics']=dict(kernel_condition=condition,amplitude_expansion=expansion,
        unstable_amplitudes=bool(condition>1e8 or expansion>1e4),
        interpretation='Diagnostic flag only: condition >1e8 or max|A|/max|Y| >1e4; estimates are not changed')
    r['legacy_error']=legacy_error_fit(b,y,r['D'],r['amplitudes'])
    return r


def fit(b,Y,ppm=None,windows=None,orders=None,config=None):
    """Fit disjoint frequency windows; windows use Python [start,stop) indices.

    Windows are automatically segmented if omitted; orders default to the
    native singular-value heuristic plus native reduction rules. No true D or
    amplitude is accepted. A whole single decay can be passed as one column.
    """
    started=time.perf_counter();cfg=config or Config();b,y=observations(b,Y);f=y.shape[1]
    ppm=np.arange(f,dtype=float) if ppm is None else np.asarray(ppm,float).ravel()
    if len(ppm)!=f or not np.isfinite(ppm).all() or (f>1 and not (np.all(np.diff(ppm)>0) or np.all(np.diff(ppm)<0))):raise ValueError('ppm must be a finite, strictly monotone frequency axis')
    windows=segment(y[0],cfg.threshold_percent,cfg.padding_points) if windows is None else np.asarray(windows)
    if windows.size==0:windows=np.empty((0,2),int)
    if windows.ndim!=2 or windows.shape[1]!=2 or not np.isfinite(windows).all() or np.any(windows!=np.floor(windows)):raise ValueError('windows must contain integer [start,stop) pairs')
    windows=windows.astype(int)
    if np.any(windows[:,0]<0) or np.any(windows[:,1]>f) or np.any(windows[:,0]>=windows[:,1]):raise ValueError('Window outside frequency range or empty')
    coverage=np.zeros(f,int)
    for a,z in windows:coverage[a:z]+=1
    if np.any(coverage>1):raise ValueError('Overlapping windows would double-count spectral amplitude')
    if orders is not None:
        raw=np.asarray(orders).ravel()
        if len(raw)!=len(windows) or not np.isfinite(raw).all() or np.any(raw!=np.floor(raw)):raise ValueError('Provide one integer order per window')
        orders=raw.astype(int)
    output=[];drows=[];arows=[];failures=[]
    for i,(a,z) in enumerate(windows):
        q=int(orders[i]) if orders is not None else estimate_order(y[:,a:z],cfg.svd_cutoff,min(cfg.max_components,z-a,len(b)-1))
        if not 1<=q<=min(cfg.max_components,z-a,len(b)-1):raise ValueError('Requested local order exceeds window dimensions')
        try:r=fit_window(b,y[:,a:z],q,cfg)
        except ValueError as exc:
            failures.append(dict(window=i,error=str(exc)));output.append(dict(window=i,start=int(a),stop=int(z),status='failed',error=str(exc)));continue
        r.update(window=i,start=int(a),stop=int(z),status='ok',initial_order=q)
        global_a=np.zeros((len(r['D']),f));global_a[:,a:z]=r['amplitudes']
        drows.extend(r['D']);arows.extend(global_a);output.append(r)
    d=np.asarray(drows);a=np.asarray(arows).reshape(len(d),f)
    prediction=kernel(b,d)@a;grid=np.linspace(0,cfg.max_diffusion,cfg.diffusion_bins);delta=grid[1]-grid[0]
    mass=np.zeros((cfg.diffusion_bins,f));legacy=np.zeros((f,cfg.diffusion_bins));outside=0.;undefined=0
    for r in output:
        if r['status']!='ok':continue
        for j,value in enumerate(r['D']):
            amp=r['amplitudes'][j];start,stop=r['start'],r['stop']
            if 0<=value<=cfg.max_diffusion:mass[min(int(np.floor(value/delta+.5)),len(grid)-1),start:stop]+=amp
            else:outside+=float(np.abs(amp).sum())
            se=r['legacy_error']['standard_error_D'][j];scale=r['legacy_error']['scales'][j]
            if not np.isfinite(se) or se<0:undefined+=1;continue
            if se==0:
                weights=np.where(abs(grid-value)<delta/2,2.,np.where(abs(grid-value)==delta/2,1.,0.))
            else:weights=erf((grid+delta/2-value)/(np.sqrt(2)*se))-erf((grid-delta/2-value)/(np.sqrt(2)*se))
            legacy[start:stop]+=amp[:,None]*scale*weights[None,:]
    return dict(D=d,amplitudes=a,prediction=prediction,windows=output,window_points=windows,
                coverage=coverage,ppm=ppm,diffusion_grid=grid,mass_grid=mass,legacy_DOSY=legacy,
                display=dict(bins=cfg.diffusion_bins,outside_absolute_mass=outside,undefined_legacy_widths=undefined,
                             interpretation='mass_grid stores discrete signed amplitudes; legacy_DOSY uses GNAT error broadening and factor 2, not a recovered continuous distribution'),
                config=asdict(cfg),failures=failures,seconds=time.perf_counter()-started,
                unstable_windows=[r['window'] for r in output if r.get('numerical_diagnostics',{}).get('unstable_amplitudes',False)],
                algorithm='LOCODOSY / GNAT Python adaptation',source_doi='10.1021/ja2004895',
                selected_rank_interpretation='Local orders only; number of output rows is not the global molecular count')


def cluster_components(D,standard_error,axis_height,multiplier=4.):
    """Headless port of locodosyclustering/sclust; axes height is explicit.

    Returns fuzzy candidate centroids and per-input influence limits; it does
    not assign components exclusively or merge amplitudes. Native clustering
    depends on the plot's axis height, so it is deliberately not called by fit.
    """
    d=np.asarray(D,float).ravel();se=np.asarray(standard_error,float).ravel()
    if not len(d) or len(d)!=len(se) or not np.isfinite(d).all() or not np.isfinite(se).all() or np.any(se<=0):raise ValueError('Finite D and positive defined errors required')
    if not np.isfinite(axis_height) or axis_height<=0 or not np.isfinite(multiplier) or multiplier<=0:raise ValueError('Positive axis_height and multiplier required')
    # Public API uses physical D; native GUI calculations use D / 1e-10.
    d=d/1e-10;se=se/1e-10;axis_height=axis_height/1e-10
    widths=abs(se/axis_height)*multiplier;lo=d.min();hi=d.max()
    if hi==lo:lo-=.0001*(1+abs(lo));hi+=.0001*(1+abs(hi))
    x=np.clip((d-lo)/(hi-lo),0,1);pot=np.exp(-4*((x[:,None]-x[None,:])/widths[None,:])**2).sum(axis=1)
    initial=pot.max();centers=[]
    for _ in range(len(d)*3):
        ix=int(np.argmax(pot));best=pot[ix]
        if best<=0:break
        ratio=best/initial;accept=ratio>.5
        if not accept:
            if ratio<=.15:break
            nearest=None
            for v in centers:
                overlap=(x[ix]-v)/widths;distance=np.outer(overlap,overlap)
                if nearest is None or np.all(distance<nearest):nearest=distance
            # MATLAB's matrix condition requires ALL entries to be true.
            accept=bool(np.all(ratio+np.sqrt(nearest)>=1))
        if accept:
            centers.append(x[ix]);pot=np.maximum(0,pot-best*np.exp(-4*((x[ix]-x)/(1.25*widths))**2))
        else:pot[ix]=0
    return (np.asarray(centers)*(hi-lo)+lo)*1e-10,widths*(hi-lo)*1e-10
