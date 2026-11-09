"""Continuous positive support recovery from moment measures.

Residual-driven births and continuous sliding are inspired by sliding
Frank--Wolfe. This finite-order, likelihood-selected hybrid is not a BLASSO
implementation and does not inherit its global convergence certificates.
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import chi2
from .components import exponential_kernel, _profile


def weighted_locations(nodes,weights,count,bounds):
    """Deterministic 1D clustering for initialization only; never final merging."""
    nodes=np.asarray(nodes);weights=np.asarray(weights)
    if weights.ndim==2:weights=weights.sum(axis=1)
    use=(weights>0)&np.isfinite(weights)&(nodes>=bounds[0])&(nodes<=bounds[1])
    if not np.any(use):return np.geomspace(*bounds,count+2)[1:-1]
    z=np.log(nodes[use]);w=weights[use];order=np.argsort(z);z,w=z[order],w[order]
    cumulative=(np.cumsum(w)-.5*w)/w.sum()
    centers=np.interp((np.arange(count)+.5)/count,cumulative,z)
    for _ in range(30):
        labels=np.argmin(abs(z[:,None]-centers),axis=1)
        new=np.array([np.average(z[labels==j],weights=w[labels==j]) if np.any(labels==j) else centers[j] for j in range(count)])
        if np.max(abs(new-centers))<1e-10:break
        centers=new
    return np.exp(np.sort(centers))


def _slide(b,y,sigma,initial,limits,max_iterations=800):
    initial=np.clip(np.sort(initial),*limits)
    opt=minimize(lambda z:_profile(z,b,y,sigma)[:2],initial,jac=True,method='L-BFGS-B',
                 bounds=[limits]*len(initial),options=dict(maxiter=max_iterations,ftol=2e-13,gtol=1e-7,maxls=50))
    z=np.sort(opt.x);value,gradient,a=_profile(z,b,y,sigma)
    projected=z-np.clip(z-gradient,*limits)
    # Backend success and independently recomputed value are both retained.
    return dict(z=z,amplitudes=a,rss=2*value,success=bool(opt.success),
                iterations=int(opt.nit),projected_gradient=float(np.max(abs(projected))),message=str(opt.message))


def birth_locations(b,residual,sigma,limits,count=3,*,current_rates=None,current_amplitudes=None,return_scores=False):
    """Maximize a positive-amplitude residual improvement oracle in log-rate.

    Profile out current positive amplitudes and cap the proposed step so
    those amplitudes remain nonnegative. The score is a feasible decrease
    in weighted RSS, rather than raw correlation with almost parallel kernels.
    A grid brackets local maxima; continuous scalar searches refine them.
    The finite search is a proposal oracle, not a global dual certificate.
    """
    nuisance=[]
    for q in range(sigma.shape[1]):
        if current_rates is None or not len(current_rates):
            matrix=np.zeros((len(b),0));amps=np.empty(0)
        else:
            active=current_amplitudes[:,q]>1e-10*max(current_amplitudes[:,q].sum(),1e-12)
            matrix=exponential_kernel(b,current_rates[active])/sigma[:,q,None]
            amps=current_amplitudes[active,q]
        nuisance.append((matrix,np.linalg.pinv(matrix,rcond=1e-12),amps))
    def score(z):
        k=np.exp(-b*np.exp(z))[:,None]/sigma
        gain=0.
        for q,(matrix,inverse,amps) in enumerate(nuisance):
            coefficients=inverse@k[:,q]
            direction=k[:,q]-matrix@coefficients
            norm=float(direction@direction)
            if norm<=1e-14*max(float(k[:,q]@k[:,q]),1e-300):continue
            correlation=float(direction@residual[:,q])
            step=max(correlation/norm,0.)
            positive=coefficients>0
            if np.any(positive):step=min(step,float(np.min(amps[positive]/coefficients[positive])))
            gain+=max(0.,2*step*correlation-step*step*norm)
        return gain
    grid=np.linspace(*limits,129);values=np.array([score(z) for z in grid])
    maxima=np.flatnonzero((values>=np.r_[-np.inf,values[:-1]])&(values>=np.r_[values[1:],-np.inf]))
    candidates=[]
    for i in maxima:
        if i==0 or i==len(grid)-1:candidates.append((values[i],grid[i]))
        else:
            opt=minimize_scalar(lambda z:-score(z),bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-10})
            candidates.append((-float(opt.fun),float(opt.x)))
    chosen=sorted(candidates,reverse=True)[:count]
    return chosen if return_scores else [z for _,z in chosen]


def rate_information(b,y,sigma,rates,amplitudes):
    """Local nuisance-projected information; explicitly not calibrated coverage."""
    k=len(rates)
    if not k:return dict(local_log_rate_sd=[],information_condition=None,precision_supported=False)
    kernel=exponential_kernel(b,rates);derivative=-b[:,None]*rates*kernel
    information=np.zeros((k,k))
    for q in range(y.shape[1]):
        a=kernel/sigma[:,q,None];active=amplitudes[:,q]>1e-8*max(amplitudes[:,q].sum(),1e-12)
        g=derivative*amplitudes[:,q]/sigma[:,q,None]
        nuisance=a[:,active]
        if nuisance.shape[1]:g-=nuisance@np.linalg.lstsq(nuisance,g,rcond=1e-12)[0]
        information+=g.T@g
    eigen=np.linalg.eigvalsh(information)
    valid=eigen[-1]>0 and eigen[0]>1e-12*eigen[-1]
    sd=np.sqrt(np.maximum(0,np.diag(np.linalg.inv(information)))) if valid else None
    return dict(local_log_rate_sd=None if sd is None else sd.tolist(),
                information_condition=float(eigen[-1]/eigen[0]) if valid else None,
                precision_supported=bool(valid and np.max(sd)<.1))


def recover_support(b,y,sigma,bounds,nodes,weights,*,max_components=8,max_iterations=800,extra_measure=None):
    """Birth, slide and delete continuous rates without a supplied truth count.

    Moment clustering seeds the candidates. Data-driven births and domain
    starts remain available if the moment representation is diffuse or a
    learned proposal is inaccurate. Model evidence is a BIC heuristic, not
    the held-out loss previously used to select the moment representation.
    """
    reference=np.sqrt(bounds[0])*np.sqrt(bounds[1]);t=np.asarray(b)*reference
    scales=np.maximum(np.max(abs(y),axis=0),np.median(sigma,axis=0))
    yn,sn=y/scales,sigma/scales
    lo,hi=np.log(np.array(bounds)/reference);limits=(lo,hi)
    cap=min(max_components,max(1,(y.size-1)//(y.shape[1]+1)))
    null=dict(z=np.empty(0),amplitudes=np.zeros((0,y.shape[1])),rss=float(np.sum((yn/sn)**2)),
              success=True,iterations=0,projected_gradient=0.,message='zero measure')
    fits={0:null};history=[]
    def score(fit):
        # Conservative full-amplitude parameter count retained at NNLS
        # boundaries; no hidden reduction of complexity for near-zero weights.
        return fit['rss']+len(fit['z'])*(y.shape[1]+1)*np.log(y.size)
    def record(count,fit,operations):
        return dict(order=count,rss=fit['rss'],criterion=score(fit),success=fit['success'],
                    rates=(np.exp(fit['z'])*reference).tolist(),iterations=fit['iterations'],
                    projected_gradient=fit['projected_gradient'],message=fit['message'],operations=operations)
    history.append(record(0,null,[]))
    for count in range(1,cap+1):
        previous=fits[count-1]
        residual=(yn-exponential_kernel(t,np.exp(previous['z']))@previous['amplitudes'])/sn
        births=birth_locations(t,residual,sn,limits,current_rates=np.exp(previous['z']),current_amplitudes=previous['amplitudes'])
        starts=[('profiled_birth',np.sort(np.r_[previous['z'],z])) for z in births]
        starts.append(('moment_seed',np.log(weighted_locations(nodes,weights,count,bounds)/reference)))
        starts.append(('domain_seed',np.linspace(lo,hi,count+2)[1:-1]))
        if extra_measure is not None:
            starts.append(('learned_seed',np.log(weighted_locations(*extra_measure,count,bounds)/reference)))
        candidates=[];operations=[]
        for label,initial in starts:
            fit=_slide(t,yn,sn,initial,limits,max_iterations)
            candidates.append(fit)
            operations.append(dict(kind=label,rss=fit['rss'],success=fit['success'],iterations=fit['iterations']))
        successful=[fit for fit in candidates if fit['success']]
        best=min(successful or candidates,key=lambda f:f['rss'])
        fits[count]=best;history.append(record(count,best,operations))
        # Delete-and-slide can recover a better lower-order basin reached by
        # a richer measure. Keep only verified decreases at that fixed order.
        if count>1 and best['success']:
            for j in range(count):
                reduced=_slide(t,yn,sn,np.delete(best['z'],j),limits,max_iterations)
                if reduced['success'] and (not fits[count-1]['success'] or reduced['rss']<fits[count-1]['rss']-1e-8):
                    fits[count-1]=reduced
                    history[count-1]=record(count-1,reduced,history[count-1]['operations']+[
                        dict(kind='delete_and_slide',parent_order=count,rss=reduced['rss'],success=True,iterations=reduced['iterations'])])
    # A final backward sweep propagates improved high-order basins through
    # multiple deletions, instead of losing them at the end of the forward path.
    for count in range(cap,1,-1):
        if not fits[count]['success']:continue
        for j in range(count):
            reduced=_slide(t,yn,sn,np.delete(fits[count]['z'],j),limits,max_iterations)
            if reduced['success'] and (not fits[count-1]['success'] or reduced['rss']<fits[count-1]['rss']-1e-8):
                fits[count-1]=reduced
                history[count-1]=record(count-1,reduced,history[count-1]['operations']+[
                    dict(kind='backward_delete_and_slide',parent_order=count,rss=reduced['rss'],success=True,iterations=reduced['iterations'])])
    valid=[(order,fit) for order,fit in fits.items() if fit['success']]
    order,best=min(valid,key=lambda item:score(item[1]))
    best_score=score(best)
    for entry in history:
        entry['selected']=entry['order']==order;entry['delta_criterion']=entry['criterion']-best_score
    rates=np.exp(best['z'])*reference;amplitudes=best['amplitudes']*scales
    dof=max(1,y.size-order*(y.shape[1]+1))
    info=rate_information(b,y,sigma,rates,amplitudes)
    ambiguity=sum(h['success'] and h['delta_criterion']<=2 for h in history)>1
    adequate=best['rss']<=chi2.ppf(.99,dof)
    bounds_hit=bool(len(rates) and np.any((rates<=bounds[0]*(1+1e-5))|(rates>=bounds[1]*(1-1e-5))))
    diagnostics=dict(optimizer_success=bool(best['success']),algorithm='moment-seeded continuous birth-slide-delete search',
        selected_components=order,selected_rss=best['rss'],selection_criterion='known-noise BIC heuristic on exact exponential predictions',
        selection_uses_all_rows=True,model_order_ambiguous=bool(ambiguity),noise_fit_adequate=bool(adequate),
        component_limit_reached=order==cap,rate_bound_hit=bounds_hit,**info,
        model_search_complete=all(fit['success'] for fit in fits.values()),
        uncertainty_interpretation='local curvature and order heuristic; not calibrated confidence or global identifiability',
        inference_conditional_on='finite positive shared-support mixture',
        optimization_global_certificate=False)
    return rates,amplitudes,history,diagnostics
