"""Strict finite exponential mixtures with residual-guided variable projection.

The exact active-set residual Jacobian and variable projection are established
methods. Rank testing uses a plug-in parametric bootstrap, not a finite-sample
guarantee. This experimental RAI variant does not claim global optimality,
chemical identification, or established scientific novelty.
"""
from dataclasses import dataclass, asdict, replace
import numpy as np
from scipy.optimize import least_squares, nnls
from scipy.stats import chi2
from .model import DecayProblem


@dataclass(frozen=True)
class DiscreteConfig:
    max_components: int = 4
    starts: int = 10
    max_nfev: int = 250
    seed: int = 97011
    selection: str = 'bootstrap'
    bootstraps: int = 39
    bootstrap_starts: int = 3
    alpha: float = .05
    ftol: float = 1e-11
    gtol: float = 1e-8
    stationarity_tolerance: float = 1e-6
    bins: int = 256

    def __post_init__(self):
        for key, minimum in [('max_components',1),('starts',1),('max_nfev',1),
                             ('seed',0),('bootstraps',0),('bootstrap_starts',1),('bins',128)]:
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<minimum:
                raise ValueError(key)
        if self.max_components>9 or self.selection not in ('bootstrap','bic','aicc'):
            raise ValueError('Unsupported component count or selection rule')
        for key in ('ftol','gtol','stationarity_tolerance'):
            if not np.isfinite(getattr(self,key)) or getattr(self,key)<=0:raise ValueError(key)
        if not np.isfinite(self.alpha) or not 0<self.alpha<.5:raise ValueError('alpha')
        if self.selection=='bootstrap' and self.bootstraps+1<1/self.alpha:
            raise ValueError('Too few bootstrap samples for the requested alpha')


class ProjectedResidual:
    """NNLS amplitudes and exact local derivative on a fixed active set."""
    def __init__(self,problem):
        self.p=problem
        self.origin=float(np.mean(np.log(problem.rate_bounds)))
        self.bounds=np.log(problem.rate_bounds)-self.origin
        self.z=None
        self.column_scale=problem.sigma[0]/problem.sigma[0,0]
        self.factorable_noise=np.allclose(problem.sigma,problem.sigma[:,0,None]*self.column_scale,rtol=1e-14,atol=0)

    def evaluate(self,z):
        z=np.asarray(z,dtype=float)
        if self.z is not None and np.array_equal(self.z,z):return
        d=np.exp(z+self.origin)
        with np.errstate(over='ignore'):
            q=self.p.b[:,None]*d
        h=np.exp(-q);dh=-np.minimum(q,745)*h
        k=len(d);n,p=self.p.y.shape
        amplitudes=np.zeros((k,p));residual=np.zeros((n,p));jac=np.zeros((n*p,k))
        if p>=4 and k<=4 and self.factorable_noise:
            amplitudes,residual,jac=self._batch(h,dh)
            self.z=z.copy();self.rates=d;self.h=h;self.dh=dh
            self.amplitudes=amplitudes;self.residual=residual;self.jacobian=jac
            return
        for j in range(p):
            H=h/self.p.sigma[:,j,None];Y=self.p.y[:,j]/self.p.sigma[:,j]
            a=nnls(H,Y,maxiter=max(500,100*k))[0]
            r=H@a-Y;active=np.flatnonzero(a>0)
            amplitudes[:,j]=a;residual[:,j]=r
            if not len(active):continue
            V=H[:,active]
            U,s,VT=np.linalg.svd(V,full_matrices=False)
            keep=s>max(V.shape)*np.finfo(float).eps*s[0]
            U=U[:,keep];pinv=(VT[keep].T/s[keep])@U.T
            local=np.zeros((n,k))
            for col,atom in enumerate(active):
                derivative=dh[:,atom]/self.p.sigma[:,j]
                v=derivative*a[atom]
                # Differentiate V.T r=0, including the residual-dependent term.
                local[:,atom]=v-U@(U.T@v)-pinv[col]*float(derivative@r)
            jac[j*n:(j+1)*n]=local
        self.z=z.copy();self.rates=d;self.h=h;self.dh=dh
        self.amplitudes=amplitudes;self.residual=residual;self.jacobian=jac

    def _batch(self,h,dh):
        """Exact small-rank NNLS by active-set enumeration, batched over spectra.

        At k<=4 only 16 faces exist. Every feasible face minimizer is examined;
        this is equivalent to independent NNLS, with shared factorizations.
        """
        n,k=h.shape;p=self.p.y.shape[1];sigma=self.p.sigma[:,0]
        H=h/sigma[:,None];Y=self.p.y/sigma[:,None];derivative=dh/sigma[:,None]
        amplitude=np.zeros((k,p));best=np.sum(Y*Y,axis=0)
        faces={}
        for mask in range(1,1<<k):
            active=np.flatnonzero(mask&(1<<np.arange(k)))
            V=H[:,active];U,s,VT=np.linalg.svd(V,full_matrices=False)
            keep=s>max(V.shape)*np.finfo(float).eps*s[0]
            U=U[:,keep];pinv=(VT[keep].T/s[keep])@U.T
            a=pinv@Y;feasible=np.all(a>=0,axis=0)
            r=V@a-Y;cost=np.sum(r*r,axis=0)
            take=feasible&(cost<best)
            amplitude[:,take]=0;amplitude[np.ix_(active,take)]=a[:,take];best[take]=cost[take]
            faces[mask]=(active,U,pinv)
        residual=H@amplitude-Y
        jac=np.zeros((n,p,k))
        labels=np.sum((amplitude>0)*(1<<np.arange(k))[:,None],axis=0)
        for mask in np.unique(labels):
            if mask==0:continue
            active,U,pinv=faces[mask];columns=np.flatnonzero(labels==mask)
            for col,atom in enumerate(active):
                v=derivative[:,atom,None]*amplitude[atom,columns]
                jac[:,columns,atom]=v-U@(U.T@v)-pinv[col,:,None]*(derivative[:,atom]@residual[:,columns])
        residual/=self.column_scale
        jac/=self.column_scale[None,:,None]
        return amplitude,residual,jac.transpose(1,0,2).reshape(n*p,k)

    def fun(self,z):self.evaluate(z);return self.residual.ravel(order='F')
    def jac(self,z):self.evaluate(z);return self.jacobian


def _diagnostics(op,z):
    op.evaluate(z);p=op.p;S=op.amplitudes
    W=op.residual/p.sigma
    grad=np.sum(op.dh*(W@S.T),axis=0)
    scale=np.maximum(1,np.sum(np.abs(op.dh)*((np.abs(p.y)/p.sigma**2)@S.T),axis=0))
    lo,hi=op.bounds
    # Projected-gradient mapping avoids misclassifying a point a few ulps
    # inside an active bound as a free parameter with a large KKT violation.
    pg=np.abs(z-np.clip(z-grad/scale,lo,hi))
    ga=op.h.T@W
    sa=np.maximum(1,np.max(np.abs(op.h.T@(p.y/p.sigma**2)),axis=0))
    residual=np.where(S>0,np.abs(ga),np.maximum(-ga,0))/sa
    scaled=float(max(np.max(pg),np.max(residual)))
    # Rate information after eliminating amplitudes, without the residual term.
    info=np.zeros((len(z),len(z)))
    for j in range(S.shape[1]):
        H=op.h/p.sigma[:,j,None];active=S[:,j]>0
        J=op.dh*S[:,j]/p.sigma[:,j,None]
        if active.any():J-=H[:,active]@np.linalg.lstsq(H[:,active],J,rcond=None)[0]
        info+=J.T@J
    eigen=np.linalg.eigvalsh(info)
    if eigen[-1]>0 and eigen[0]>1e-12*eigen[-1]:
        covariance=np.linalg.inv(info);sd=np.sqrt(np.maximum(0,np.diag(covariance)))
        separation=[]
        for i in range(len(z)-1):
            variance=max(0,covariance[i,i]+covariance[i+1,i+1]-2*covariance[i,i+1])
            separation.append(float((z[i+1]-z[i])/max(np.sqrt(variance),1e-300)))
        sd=sd.tolist();condition=float(eigen[-1]/eigen[0])
    else:sd=None;separation=None;condition=None
    return dict(scaled_kkt_residual=scaled,local_log_rate_sd=sd,
        local_separation_z=separation,information_condition=condition,
        rate_bound_hit=bool(np.any((z<=lo+1e-6)|(z>=hi-1e-6))),
        precision_interpretation='Local curvature conditional on rank; not confidence intervals or a resolution certificate')


def _starts(op,k,config,previous=None,warm=None):
    lo,hi=op.bounds
    seeds=[]
    if warm is not None and len(warm)==k:seeds.append(np.log(warm)-op.origin)
    if previous is not None and len(previous['rates'])==k-1 and k>1:
        old=np.log(previous['rates'])-op.origin
        grid=np.linspace(lo+.001,hi-.001,65)
        # Exact conditional NNLS scores propose a missing atom without truth.
        scored=[]
        for z in grid:
            trial=np.sort(np.r_[old,z]);r=op.fun(trial)
            scored.append((float(r@r),trial))
        seeds.extend(v for _,v in sorted(scored,key=lambda t:t[0])[:2])
        # Split a fitted atom to cross a coincident-rate saddle/flat region.
        for index in np.argsort(-np.sum(previous['amplitudes'],axis=1))[:2]:
            seed=np.r_[np.delete(old,index),old[index]-.18,old[index]+.18]
            seeds.append(np.sort(seed))
    seeds.append(np.linspace(lo,hi,k+2)[1:-1])
    rng=np.random.default_rng(config.seed+1009*k)
    while len(seeds)<config.starts:
        seeds.append(np.sort(rng.uniform(lo,hi,k)))
    return [np.clip(v,lo+1e-9,hi-1e-9) for v in seeds[:config.starts]]


def fit_rank(problem,k,config=DiscreteConfig(),previous=None,warm=None):
    """Fit a supplied rank. Truth may supply k only in explicitly labelled controls."""
    if isinstance(k,bool) or not isinstance(k,(int,np.integer)) or not 0<=k<=config.max_components:
        raise ValueError('rank')
    if k==0:
        return dict(rank=0,rates=np.empty(0),amplitudes=np.zeros((0,problem.y.shape[1])),
            prediction=np.zeros_like(problem.y),rss=float(np.sum((problem.y/problem.sigma)**2)),
            native_success=True,locally_stationary=True,scaled_kkt_residual=0.,starts=[])
    if problem.y.size<=k*(problem.y.shape[1]+1):raise ValueError('Too few observations for requested rank')
    op=ProjectedResidual(problem);lo,hi=op.bounds
    starts=_starts(op,k,config,previous,warm);outcomes=[];solutions=[]
    for z in starts:
        initial_r=op.fun(z);initial_rss=float(initial_r@initial_r)
        try:
            opt=least_squares(op.fun,z,jac=op.jac,bounds=(lo,hi),method='trf',
                ftol=config.ftol,xtol=config.ftol,gtol=config.gtol,max_nfev=config.max_nfev)
            selected=opt.x if 2*opt.cost<=initial_rss else z
            op.evaluate(selected);rss=float(np.sum(op.residual**2))
            outcomes.append(dict(rss=rss,initial_rss=initial_rss,native_success=bool(opt.success),
                status=int(opt.status),message=str(opt.message),nfev=int(opt.nfev),
                initial_retained=bool(2*opt.cost>initial_rss)))
            solutions.append(selected.copy())
        except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:
            outcomes.append(dict(native_success=False,error=str(exc),rss=None))
            solutions.append(None)
    eligible=[i for i,z in enumerate(solutions) if z is not None]
    if not eligible:raise RuntimeError('All fixed-rank fits failed')
    best=min(eligible,key=lambda i:outcomes[i]['rss'])
    z=np.sort(solutions[best]);diag=_diagnostics(op,z)
    polish=None
    if diag['scaled_kkt_residual']>config.stationarity_tolerance:
        before=float(np.sum(op.residual**2))
        improved=least_squares(op.fun,z,jac=op.jac,bounds=(lo,hi),method='trf',
            ftol=None,xtol=1e-14,gtol=min(config.gtol,1e-10),max_nfev=config.max_nfev)
        # Permit only roundoff-sized objective differences when polishing a
        # stationarity failure; never silently discard a failed refinement.
        accepted=2*improved.cost<=before+1e-12*max(1,before)
        if accepted:z=np.sort(improved.x)
        polish=dict(native_success=bool(improved.success),status=int(improved.status),
            message=str(improved.message),nfev=int(improved.nfev),accepted=accepted,
            initial_rss=before,final_rss=float(2*improved.cost))
        diag=_diagnostics(op,z)
    return dict(rank=k,rates=op.rates.copy(),amplitudes=op.amplitudes.copy(),
        prediction=op.h@op.amplitudes,rss=float(np.sum(op.residual**2)),
        native_success=outcomes[best]['native_success'],starts=outcomes,polish=polish,
        locally_stationary=diag['scaled_kkt_residual']<=config.stationarity_tolerance,**diag)


def rank_path(problem,config=DiscreteConfig()):
    fits=[fit_rank(problem,0,config)]
    for k in range(1,config.max_components+1):
        fits.append(fit_rank(problem,k,config,previous=fits[-1]))
    return fits


def bootstrap_rank(problem,fits,config):
    """Sequential plug-in likelihood-ratio tests with refitting in every replicate.

    Monte Carlo p-values include the observed statistic. They are approximate
    because nuisance parameters are fitted; mixtures are nonregular. Failed
    replicate optimization produces an unresolved decision, not acceptance.
    """
    selected=0;tests=[]
    small=replace(config,starts=config.bootstrap_starts,selection='bic',bootstraps=0)
    rng=np.random.default_rng(config.seed+81013)
    for k in range(config.max_components):
        null,alternative=fits[k:k+2]
        delta=max(0.,null['rss']-alternative['rss'])
        stats=[];failed=0
        for _ in range(config.bootstraps):
            y=null['prediction']+problem.sigma*rng.normal(size=problem.y.shape)
            p=DecayProblem(problem.b,y,problem.sigma,problem.rate_bounds,problem.ppm)
            try:
                f0=fit_rank(p,k,small,warm=null['rates'])
                f1=fit_rank(p,k+1,small,previous=f0,warm=alternative['rates'])
                if not f0['locally_stationary'] or not f1['locally_stationary']:
                    failed+=1
                stats.append(max(0.,f0['rss']-f1['rss']))
            except (RuntimeError,ValueError,np.linalg.LinAlgError):failed+=1
        if failed or len(stats)!=config.bootstraps or not null['locally_stationary'] or not alternative['locally_stationary']:
            tests.append(dict(null_rank=k,alternative_rank=k+1,delta_rss=delta,failed_replicates=failed,
                status='unresolved_optimization',pvalue=None,bootstrap_statistics=stats))
            return selected,tests,False
        pvalue=(1+sum(v>=delta for v in stats))/(config.bootstraps+1)
        reject=pvalue<=config.alpha
        tests.append(dict(null_rank=k,alternative_rank=k+1,delta_rss=delta,failed_replicates=0,
            status='reject_null' if reject else 'retain_null',pvalue=float(pvalue),
            monte_carlo_standard_error=float(np.sqrt(pvalue*(1-pvalue)/(config.bootstraps+1))),
            bootstrap_statistics=stats))
        if not reject:return selected,tests,True
        selected=k+1
    return selected,tests,True


@dataclass
class DiscreteResult:
    rates: np.ndarray
    amplitudes: np.ndarray
    reconstructed: np.ndarray
    rate_edges: np.ndarray
    masses: np.ndarray
    candidates: list
    diagnostics: dict
    config: dict

    def predict(self,b):return np.exp(-np.asarray(b)[:,None]*self.rates)@self.amplitudes


def solve_discrete(problem,config=DiscreteConfig(),known_components=None):
    """Single curve or joint DOSY; all profiles are finite point measures."""
    if problem.y.shape[1]>1 and problem.ppm is None:
        raise ValueError('Multiple curves require an explicit frequency coordinate')
    # Canonical frequency ordering makes the floating-point fit reproducible.
    order=np.argsort(problem.ppm,kind='stable') if problem.ppm is not None else np.arange(problem.y.shape[1])
    back=np.argsort(order)
    p=DecayProblem(problem.b,problem.y[:,order],problem.sigma[:,order],problem.rate_bounds,
                   problem.ppm[order] if problem.ppm is not None else None)
    if known_components is not None:
        fits=[fit_rank(p,known_components,config)];chosen=0;tests=[];selection_valid=True
    else:
        fits=rank_path(p,config);tests=[];selection_valid=True
        n=p.y.size;signals=p.y.shape[1]
        for f in fits:
            df=f['rank']*(signals+1)
            f['bic']=float(f['rss']+df*np.log(n))
            f['aicc']=float(f['rss']+2*df+2*df*(df+1)/(n-df-1)) if n>df+1 else None
        if config.selection=='bootstrap':
            chosen,tests,selection_valid=bootstrap_rank(p,fits,config)
        else:
            eligible=[i for i,f in enumerate(fits) if f[config.selection] is not None]
            chosen=min(eligible,key=lambda i:fits[i][config.selection])
    best=fits[chosen];edges=np.geomspace(*p.rate_bounds,config.bins+1)
    masses=np.zeros((config.bins,p.y.shape[1]))
    for d,a in zip(best['rates'],best['amplitudes']):
        index=np.clip(np.searchsorted(edges,d,side='right')-1,0,config.bins-1);masses[index]+=a
    candidates=[]
    for i,f in enumerate(fits):
        row={key:(value.tolist() if isinstance(value,np.ndarray) else value)
             for key,value in f.items() if key!='prediction'}
        row['amplitudes']=f['amplitudes'][:,back].tolist();row['selected']=i==chosen
        candidates.append(row)
    k=best['rank'];dof=max(1,p.y.size-k*(p.y.shape[1]+1))
    diagnostic=dict(selected_components=k,known_components_supplied=known_components is not None,
        selection='supplied rank' if known_components is not None else config.selection,
        selection_valid=selection_valid,bootstrap_tests=tests,
        native_success=best['native_success'],locally_stationary=best['locally_stationary'],
        scaled_kkt_residual=best['scaled_kkt_residual'],
        rank_limit_reached=k==config.max_components,
        noise_fit_adequate=bool(best['rss']<=chi2.ppf(.99,dof)),
        noise_adequacy_interpretation='Nominal known-Gaussian-noise check; not selection-adjusted',
        global_optimum_certified=False,chemical_species_identified=False,
        scientific_novelty_established=False,
        search_complete=all(f['locally_stationary'] for f in fits),
        local_log_rate_sd=best.get('local_log_rate_sd'),local_separation_z=best.get('local_separation_z'),
        rate_bound_hit=best.get('rate_bound_hit',False),
        uncertainty_interpretation='Conditional local curvature and plug-in bootstrap; empirical calibration required')
    return DiscreteResult(best['rates'],best['amplitudes'][:,back],best['prediction'][:,back],
                          edges,masses[:,back],candidates,diagnostic,asdict(config))
