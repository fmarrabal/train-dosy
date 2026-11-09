"""Experimental mixed-measure unmixing with ordinary exp(-b D) attenuation.

DOSY denotes shared physical factors across spectral columns, not a density
type. Factors may be atomic or continuous. Lognormal profile mixtures are a
restricted approximation class, not arbitrary-density or chemical uniqueness.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
import numpy as np
from scipy.optimize import minimize, nnls
from scipy.special import ndtr, log_ndtr
from scipy.stats import chi2
from .model import DecayProblem, real_array


@dataclass(frozen=True)
class MixedConfig:
    max_factors: int = 3
    starts: int = 4
    seed: int = 86701
    bins: int = 256
    quadrature_order: int = 256
    min_log_width: float = .035
    max_log_width: float = 1.2
    max_iterations: int = 500
    ftol: float = 1e-11
    gtol: float = 1e-6
    selection: str = 'discrepancy_bic'
    adequacy_probability: float = .95
    skewed_profiles: bool = True
    measure: str = 'mixed'

    def __post_init__(self):
        for name,lo in [('max_factors',1),('starts',1),('seed',0),('bins',128),('quadrature_order',128),('max_iterations',1)]:
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<lo:
                raise ValueError(f'{name} must be an integer >= {lo}')
        if not 0<self.min_log_width<self.max_log_width or not np.isfinite(self.max_log_width):
            raise ValueError('Need finite ordered positive width bounds')
        if not np.isfinite(self.ftol) or not np.isfinite(self.gtol) or min(self.ftol,self.gtol)<=0:
            raise ValueError('Optimization tolerances must be positive')
        if self.selection!='discrepancy_bic':raise ValueError('Only discrepancy then BIC is implemented')
        if not 0<self.adequacy_probability<1:raise ValueError('Invalid adequacy probability')
        if self.measure not in ('mixed','continuous','discrete'):raise ValueError('Invalid measure class')


@lru_cache(maxsize=8)
def _quadrature(order):
    return np.polynomial.legendre.leggauss(order)


class ProfileOperator:
    """Atomic kernels and normalized, truncated Gaussian densities in log D."""
    def __init__(self,b,bounds,order=256,skewed=False):
        self.b=real_array(b,'b').reshape(-1)
        bounds=real_array(bounds,'rate_bounds').reshape(-1)
        if len(bounds)!=2 or not 0<bounds[0]<bounds[1] or np.any(self.b<0):
            raise ValueError('Invalid diffusion domain or acquisition values')
        self.origin=float(np.mean(np.log(bounds)))
        self.lo,self.hi=np.log(bounds)-self.origin
        q,w=_quadrature(order)
        self.nodes=(self.hi+self.lo)/2+(self.hi-self.lo)*q/2
        self.weights=w*(self.hi-self.lo)/2
        self.kernel=np.exp(-self.b[:,None]*np.exp(self.nodes+self.origin))
        self.skewed=skewed
        self.order=order

    def evaluate(self,theta,rank,broad):
        centers=theta[:rank]
        widths=np.exp(theta[rank:rank+broad])
        h=np.empty((len(self.b),rank));dm=np.empty_like(h);dw=np.empty((len(self.b),broad*(2 if self.skewed else 1)))
        if broad:
            if self.skewed:
                # Split at profile-centered scales so narrow skew transitions
                # are resolved without requiring an enormous fixed dictionary.
                q,w=_quadrature(max(16,int(np.ceil(self.order/8))))
                for k in range(broad):
                    m=centers[k];s=widths[k];alpha=theta[rank+broad+k]
                    cuts=np.unique(np.clip(np.r_[self.lo,self.hi,m+s*np.array([-8,-3,-1,0,1,3,8])],self.lo,self.hi))
                    nodes=((cuts[:-1,None]+cuts[1:,None])/2+np.diff(cuts)[:,None]*q/2).ravel()
                    weight=(np.diff(cuts)[:,None]*w/2).ravel()
                    u=(nodes-m)/s;v=u*alpha;logphi=log_ndtr(v)
                    mills=np.exp(-.5*v*v-.5*np.log(2*np.pi)-logphi)
                    logp=-.5*u*u+logphi;prob=np.exp(logp-logp.max())*weight;prob/=prob.sum()
                    kernel=np.exp(-self.b[:,None]*np.exp(nodes+self.origin))
                    sm=(u-alpha*mills)/s;sw=u*u-v*mills;sa=u*mills
                    h[:,k]=kernel@prob;dm[:,k]=kernel@(prob*(sm-prob@sm))
                    dw[:,k]=kernel@(prob*(sw-prob@sw));dw[:,broad+k]=kernel@(prob*(sa-prob@sa))
            else:
                self._normal(centers,widths,broad,h,dm,dw)
        if broad<rank:
            t=self.b[:,None]*np.exp(centers[None,broad:]+self.origin)
            h[:,broad:]=np.exp(-t);dm[:,broad:]=-t*h[:,broad:]
        return h,dm,dw

    def _normal(self,centers,widths,broad,h,dm,dw):
        delta=self.nodes[:,None]-centers[None,:broad]
        u=delta/widths
        logp=-.5*u**2
        sm=delta/widths**2;sw=u**2
        p=np.exp(logp-logp.max(axis=0))*self.weights[:,None]
        p/=p.sum(axis=0)
        h[:,:broad]=self.kernel@p
        dm[:,:broad]=self.kernel@(p*(sm-np.sum(p*sm,axis=0)))
        dw[:,:broad]=self.kernel@(p*(sw-np.sum(p*sw,axis=0)))


def _fit_amplitudes(h,y,sigma):
    return np.column_stack([nnls(h/sigma[:,j,None],y[:,j]/sigma[:,j],maxiter=max(100,20*h.shape[1]))[0]
                            for j in range(y.shape[1])])


def _objective(theta,op,y,sigma,rank,broad,details=False):
    h,dm,dw=op.evaluate(theta,rank,broad)
    amplitude=_fit_amplitudes(h,y,sigma)
    residual=(h@amplitude-y)/sigma
    gradient=np.r_[np.sum((dm.T@(residual/sigma))*amplitude,axis=1),
                   np.sum((dw.T@(residual/sigma))*np.tile(amplitude[:broad],(2 if op.skewed else 1,1)),axis=1)]
    value=.5*float(np.sum(residual**2))
    return (value,gradient,h,amplitude) if details else (value,gradient)


@dataclass
class MixedResult:
    centers: np.ndarray
    log_widths: np.ndarray
    is_continuous: np.ndarray
    skewness: np.ndarray
    spectra: np.ndarray
    rate_edges: np.ndarray
    continuous_masses: np.ndarray
    atomic_masses: np.ndarray
    density_log_rate: np.ndarray
    profile_masses: np.ndarray
    reconstructed: np.ndarray
    residual: np.ndarray
    fractions: np.ndarray
    diagnostics: dict
    candidates: list
    config: dict
    rate_bounds: tuple

    @property
    def masses(self):return self.continuous_masses+self.atomic_masses

    @property
    def median_rates(self):
        values=self.centers.copy()
        for k in np.flatnonzero(self.is_continuous):
            values[k]=np.exp(np.interp(.5,np.r_[0,np.cumsum(self.profile_masses[:,k])],np.log(self.rate_edges)))
        return values

    def predict(self,b):
        op=ProfileOperator(b,self.rate_bounds,self.config['quadrature_order'],self.config['skewed_profiles'])
        order=np.r_[np.flatnonzero(self.is_continuous),np.flatnonzero(~self.is_continuous)]
        theta=np.r_[np.log(self.centers[order])-op.origin,np.log(self.log_widths[order][self.is_continuous[order]])]
        if op.skewed:theta=np.r_[theta,self.skewness[order][self.is_continuous[order]]]
        return op.evaluate(theta,len(order),int(self.is_continuous.sum()))[0]@self.spectra[order]


def _recover(problem,config,joint):
    op=ProfileOperator(problem.b,problem.rate_bounds,config.quadrature_order,config.skewed_profiles)
    y=problem.y;sigma=problem.sigma;n,p=y.shape
    if not joint and p!=1:raise ValueError('Use solve_mixed_dosy for a matrix of spectral columns')
    maxrank=min(config.max_factors,max(1,(n-2)//3))
    rng=np.random.default_rng(config.seed)
    candidates=[dict(rank=0,broad=0,atomic=0,parameters=0,rss=float(np.sum((y/sigma)**2)),
                     bic=float(np.sum((y/sigma)**2)),converged=True,starts_converged=1)]
    fits=[(np.empty(0),np.empty((0,p)),np.empty((n,0)))];previous={}
    for rank in range(1,maxrank+1):
        for broad in range(rank+1):
            if config.measure=='continuous' and broad!=rank:continue
            if config.measure=='discrete' and broad!=0:continue
            bounds=[(op.lo,op.hi)]*rank+[(np.log(config.min_log_width),np.log(config.max_log_width))]*broad
            if op.skewed:bounds+=[(-10.,10.)]*broad
            initial=[]
            # Multiple deterministic starts include different broad/atomic assignments.
            even=np.linspace(op.lo,op.hi,rank+2)[1:-1]
            for start in range(config.starts):
                centers=even.copy() if start<rank else rng.uniform(op.lo,op.hi,rank)
                if start<rank:centers=np.roll(centers,start)
                widths=np.full(broad,np.log([.18,.4,.7][start%3]))
                if broad>0 and (rank,broad-1) in previous and start==0:
                    prior=previous[rank,broad-1];centers=prior[:rank].copy()
                    if broad==1:centers=np.sort(centers)
                    widths[:broad-1]=prior[rank:rank+broad-1]
                trial=np.r_[centers,widths]
                if op.skewed:trial=np.r_[trial,np.full(broad,[3,-3,0,7,-7,1][start%6])]
                initial.append(trial)
            optimized=[]
            for x0 in initial:
                fit=minimize(_objective,x0,args=(op,y,sigma,rank,broad),jac=True,method='L-BFGS-B',bounds=bounds,
                             options=dict(maxiter=config.max_iterations,ftol=config.ftol,gtol=config.gtol,maxls=40))
                optimized.append(fit)
            best=min(optimized,key=lambda f:f.fun)
            close=[f for f in optimized if f.success and f.fun<=best.fun+1e-7*max(1,abs(best.fun))]
            if close:best=min(close,key=lambda f:f.fun)
            previous[rank,broad]=best.x.copy()
            value,gradient,h,amplitude=_objective(best.x,op,y,sigma,rank,broad,True)
            # Full amplitude count is conservative; do not exploit zero coefficients to lower BIC.
            parameters=rank*p+rank+broad*(2 if op.skewed else 1)
            score=2*value+parameters*np.log(n*p)
            pg=best.x-np.clip(best.x-gradient,np.array(bounds)[:,0],np.array(bounds)[:,1])
            candidates.append(dict(rank=rank,broad=broad,atomic=rank-broad,parameters=parameters,rss=2*value,bic=float(score),
                                   converged=bool(best.success),starts_converged=sum(bool(f.success) for f in optimized),
                                   locations_unsorted=np.exp(best.x[:rank]+op.origin).tolist(),
                                   log_widths_unsorted=np.r_[np.exp(best.x[rank:rank+broad]),np.zeros(rank-broad)].tolist(),
                                   skewness_unsorted=(np.r_[best.x[rank+broad:],np.zeros(rank-broad)] if op.skewed else np.zeros(rank)).tolist(),
                                   spectra_unsorted=amplitude.tolist(),
                                   projected_gradient_inf=float(np.max(abs(pg))),iterations=int(best.nit),message=str(best.message)))
            fits.append((best.x.copy(),amplitude,h))
    # Exclude failed local solves from selection; retain all failures in the record.
    eligible=[i for i,c in enumerate(candidates) if c['converged']]
    for candidate in candidates:
        candidate['adequacy_threshold']=float(chi2.ppf(config.adequacy_probability,max(1,n*p-candidate['parameters'])))
        candidate['noise_adequate']=candidate['rss']<=candidate['adequacy_threshold']
    adequate=[i for i in eligible if candidates[i]['noise_adequate']]
    # Prefer the lowest number of shared physical profiles consistent with noise.
    # A broad profile need not be split into several atoms merely to fit noise.
    if adequate:
        minimum_rank=min(candidates[i]['rank'] for i in adequate)
        eligible=[i for i in adequate if candidates[i]['rank']==minimum_rank]
    selected=min(eligible,key=lambda i:candidates[i]['bic'])
    best=candidates[selected];theta,amplitude,h=fits[selected];rank=best['rank'];broad=best['broad']
    continuous=np.arange(rank)<broad
    centers=np.exp(theta[:rank]+op.origin);widths=np.r_[np.exp(theta[rank:rank+broad]),np.zeros(rank-broad)]
    alpha=np.r_[theta[rank+broad:],np.zeros(rank-broad)] if op.skewed else np.zeros(rank)
    order=np.argsort(centers);centers=centers[order];widths=widths[order];alpha=alpha[order];continuous=continuous[order];amplitude=amplitude[order];h=h[:,order]
    edges=np.geomspace(*problem.rate_bounds,config.bins+1);z=np.log(edges)-op.origin
    profile_mass=np.zeros((config.bins,rank));density=np.zeros_like(profile_mass)
    for k in range(rank):
        m=np.log(centers[k])-op.origin
        if continuous[k]:
            width=widths[k]
            if op.skewed:
                # Integrate each output bin; stable for skew and narrow densities.
                nodes,weights=_quadrature(12)
                zz=(z[:-1,None]+z[1:,None])/2+np.diff(z)[:,None]*nodes/2
                u=(zz-m)/width
                logp=-.5*u*u+log_ndtr(alpha[k]*u)
                profile_mass[:,k]=np.sum(np.exp(logp)*weights,axis=1)*np.diff(z)/2
                norm=profile_mass[:,k].sum();profile_mass[:,k]/=norm
            else:
                cdf=ndtr((z-m)/width);norm=(cdf[-1]-cdf[0])*width*np.sqrt(2*np.pi)
                profile_mass[:,k]=np.diff(cdf)/(cdf[-1]-cdf[0])
            mid=(z[:-1]+z[1:])/2
            density[:,k]=np.exp(-.5*((mid-m)/width)**2+(log_ndtr(alpha[k]*(mid-m)/width) if op.skewed else 0))/norm
        else:
            j=int(np.clip(np.searchsorted(edges,centers[k],side='right')-1,0,config.bins-1));profile_mass[j,k]=1.
    reconstructed=h@amplitude
    fractions=np.divide(amplitude,amplitude.sum(axis=0,keepdims=True),out=np.zeros_like(amplitude),where=amplitude.sum(axis=0,keepdims=True)>0)
    for candidate in candidates:
        candidate['delta_bic']=candidate['bic']-best['bic'];candidate['selected']=candidate is best
    dof=n*p-best['parameters']
    # Local information after eliminating active amplitudes. Conditional diagnostic only.
    information=np.zeros((len(theta),len(theta)))
    if rank:
        _,dm,dw=op.evaluate(theta,rank,broad);rawamp=fits[selected][1];rawh=fits[selected][2]
        for j in range(p):
            derivatives=np.column_stack([dm*rawamp[:,j],dw*np.tile(rawamp[:broad,j],2 if op.skewed else 1)])/sigma[:,j,None]
            nuisance=rawh[:,rawamp[:,j]>0]/sigma[:,j,None]
            derivatives-=nuisance@np.linalg.lstsq(nuisance,derivatives,rcond=1e-12)[0]
            information+=derivatives.T@derivatives
        eigen=np.linalg.eigvalsh(information)
        condition=float(eigen[-1]/eigen[0]) if eigen[0]>1e-12*eigen[-1] else None
        local_sd=np.sqrt(np.maximum(0,np.diag(np.linalg.inv(information)))).tolist() if condition is not None else None
    else:condition=None;local_sd=[]
    diag=dict(method='RAI mixed DOSY experimental' if joint else 'RAI mixed 1D experimental',converged=best['converged'],
              rank=rank,continuous_factors=broad,atomic_factors=rank-broad,
              selector='minimum noise-adequate rank, then BIC among types; thresholds heuristic for nonregular mixtures',
              selected_bic=best['bic'],order_ambiguous=sum(c['converged'] and c['delta_bic']<2 for c in candidates)>1,
              measure_type_ambiguous=any(c['converged'] and c['noise_adequate'] and c['rank']==rank and c['broad']!=broad for c in candidates),
              selector_disagrees_with_minimum_bic=best['bic']>min(c['bic'] for c in candidates if c['converged'])+1e-8,
              search_complete=all(c['converged'] for c in candidates),rank_limit_reached=rank==maxrank,
              selection_noise_adequate=best['noise_adequate'],
              reduced_chi_square=best['rss']/max(1,dof),noise_fit_adequate=bool(best['rss']<=dof+3*np.sqrt(2*max(1,dof))),
              local_information_condition=condition,local_parameter_sd_unsorted=local_sd,
              local_precision_interpretation='conditional curvature in optimization order; not calibrated intervals',
              width_boundary_hit=bool(broad and np.any(widths[continuous]<=config.min_log_width*(1+1e-4))),
              spectral_dimension_used=joint and p>1,experimental_acquisition_changes=False,
              fixed_nuisance_references_used=False,profile_class='truncated log-skew-normal mixtures plus point masses' if op.skewed else 'truncated lognormal mixtures plus point masses',
              chemical_identity_established=False,global_optimality_established=False)
    return MixedResult(centers,widths,continuous,alpha,amplitude,edges,profile_mass[:,continuous]@amplitude[continuous],
                       profile_mass[:,~continuous]@amplitude[~continuous],density@amplitude,profile_mass,
                       reconstructed,y-reconstructed,fractions,diag,candidates,asdict(config),problem.rate_bounds)


def solve_mixed_single(problem:DecayProblem,config=MixedConfig()):
    """One normal attenuation curve: discrete, continuous or mixed measure."""
    return _recover(problem,config,False)


def solve_mixed_dosy(problem:DecayProblem,config=MixedConfig()):
    """Joint b × frequency inversion with shared profiles and separate spectra."""
    return _recover(problem,config,True)
