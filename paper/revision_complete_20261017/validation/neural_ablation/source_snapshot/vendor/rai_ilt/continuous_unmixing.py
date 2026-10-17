"""Experimental continuous-profile unmixing; not a replacement for released RAI.

Single-curve inversion uses a positive cubic density and a discrepancy path.
DOSY adds group nuclear norms across compatible, possibly distant columns.
Optional cancellation profiles MUST be independently supplied references.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import cho_factor, cho_solve, null_space
from scipy.optimize import lsq_linear
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import chi2
from .model import DecayProblem, real_array


@dataclass(frozen=True)
class ContinuousConfig:
    bins: int = 256
    basis_size: int = 32
    quadrature_order: int = 8
    smoothness: float | None = None
    discrepancy_ratio: float = 1.0
    low_rank: float = 1.0
    group_probability: float = 0.95
    max_iterations: int = 12000
    tolerance: float = 2e-6

    def __post_init__(self):
        for name, minimum in (("bins", 16), ("basis_size", 6), ("quadrature_order", 4), ("max_iterations", 1)):
            v = getattr(self, name)
            if not isinstance(v, (int, np.integer)) or v < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.basis_size > self.bins:
            raise ValueError("basis_size must not exceed bins")
        for name in ("discrepancy_ratio", "tolerance"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.low_rank) or self.low_rank < 0:
            raise ValueError("low_rank must be finite and nonnegative")
        if self.smoothness is not None and (not np.isfinite(self.smoothness) or self.smoothness < 0):
            raise ValueError("smoothness must be finite and nonnegative")
        if not 0 < self.group_probability < 1:
            raise ValueError("group_probability must be between zero and one")


@dataclass
class ContinuousResult:
    rate_edges: np.ndarray
    rates: np.ndarray
    masses: np.ndarray
    density_log_rate: np.ndarray
    coefficients: np.ndarray
    reconstructed: np.ndarray
    residual: np.ndarray
    cancelled_signal: np.ndarray
    nuisance_amplitudes: np.ndarray
    groups: np.ndarray
    diagnostics: dict
    config: dict


def spline_operator(b, bounds, config=ContinuousConfig()):
    """Integrate exp(-bD) times cubic splines in z=log(D / geometric center).

    Coefficients parameterize a density per unit log D, not point masses.
    Eight-point quadrature is split at BOTH knots and reporting-bin edges.
    """
    b = real_array(b, "b").reshape(-1)
    bounds = real_array(bounds, "bounds").reshape(-1)
    if len(bounds) != 2 or not 0 < bounds[0] < bounds[1] or np.any(b < 0):
        raise ValueError("Need positive ordered bounds and nonnegative b")
    origin = (np.log(bounds[0]) + np.log(bounds[1])) / 2
    edges_z = np.linspace(np.log(bounds[0])-origin, np.log(bounds[1])-origin, config.bins+1)
    internal = np.linspace(edges_z[0], edges_z[-1], config.basis_size-2)
    knots = np.r_[np.repeat(internal[0], 3), internal, np.repeat(internal[-1], 3)]
    spline = BSpline(knots, np.eye(config.basis_size), 3)
    cuts = np.unique(np.r_[edges_z, internal])
    q, w = np.polynomial.legendre.leggauss(config.quadrature_order)
    nodes = ((cuts[:-1,None]+cuts[1:,None])/2 + np.diff(cuts)[:,None]*q/2).ravel()
    weights = (np.diff(cuts)[:,None]*w/2).ravel()
    basis = spline(nodes)
    operator = np.exp(-b[:,None]*np.exp(nodes+origin)) @ (basis*weights[:,None])
    roughness = spline.derivative(2)(nodes)*np.sqrt(weights[:,None])
    primitive = spline.antiderivative()
    mass_basis = np.diff(primitive(edges_z), axis=0)
    if mass_basis.min() < -1e-12:
        raise FloatingPointError("Spline integral lost positivity")
    mass_basis = np.maximum(mass_basis,0)  # Remove antiderivative roundoff only.
    return dict(operator=operator, roughness=roughness, mass_basis=mass_basis,
                density_basis=spline((edges_z[:-1]+edges_z[1:])/2),
                edges=np.exp(edges_z+origin), rates=np.exp((edges_z[:-1]+edges_z[1:])/2+origin))


def compatible_groups(y, sigma, probability=0.95, nuisance=None):
    """Heuristic complete-link groups of noise-compatible normalized decays.

    This is NOT a calibrated hypothesis test: scales are estimated from y.
    Complete linkage avoids chaining visibly incompatible endpoints. Chemical
    shift adjacency is deliberately not required; remote peaks can correlate.
    """
    p = y.shape[1]
    if p == 1:
        return np.zeros(1, dtype=int), np.ones(1)
    nuisance = np.empty((len(y),0)) if nuisance is None else nuisance
    scales = np.maximum(np.max(y, axis=0), 5*np.max(sigma, axis=0))
    if nuisance.shape[1]:
        for j in range(p):
            u = null_space((nuisance/sigma[:,j,None]).T)
            scales[j] = max(np.linalg.norm(u.T@(y[:,j]/sigma[:,j]))*np.median(sigma[:,j]),5*np.max(sigma[:,j]))
    normalized = y/scales
    noise = sigma/scales
    distance = np.zeros((p,p))
    for j in range(p):
        for k in range(j):
            sd = np.sqrt(noise[:,j]**2+noise[:,k]**2)
            contrast = (normalized[:,j]-normalized[:,k])/sd
            if nuisance.shape[1]:
                contrast = null_space((nuisance/sd[:,None]).T).T@contrast
            distance[j,k] = distance[k,j] = np.sum(contrast**2)
    labels = fcluster(linkage(squareform(distance), method="complete"),
                      chi2.ppf(probability, max(1,y.shape[0]-1-nuisance.shape[1])), criterion="distance")-1
    return labels, scales


def _prepare(problem, config, nuisance_profiles, nuisance_bounds, reference_source):
    basis = spline_operator(problem.b, problem.rate_bounds, config)
    n,p = problem.y.shape
    h = np.empty((n,0)) if nuisance_profiles is None else real_array(nuisance_profiles,"nuisance_profiles")
    if h.ndim == 1:
        h = h[:,None]
    if h.ndim != 2 or h.shape[0] != n:
        raise ValueError("nuisance_profiles must have shape (acquisitions, references)")
    if h.shape[1]:
        if not reference_source or nuisance_bounds is None:
            raise ValueError("Cancellation requires independent reference_source and nuisance_bounds")
        bounds = real_array(nuisance_bounds,"nuisance_bounds").reshape(-1,2)
        if bounds.shape[0] != h.shape[1] or np.any(bounds[:,0] <= 0) or np.any(bounds[:,1] <= bounds[:,0]):
            raise ValueError("Each reference requires positive ordered support bounds")
        lo,hi = problem.rate_bounds
        if np.any((bounds[:,0] < hi) & (bounds[:,1] > lo)):
            raise ValueError("Target and nuisance supports overlap: decomposition is not identified by this API")
        if h.shape[1] >= n-2:
            raise ValueError("Too many reference constraints for the acquisitions")
    matrices,observations,projections,survival = [],[],[],[]
    for j in range(p):
        hw = h/problem.sigma[:,j,None]
        if h.shape[1] and np.linalg.matrix_rank(hw) != h.shape[1]:
            raise ValueError("Reference profiles are linearly dependent")
        # Orthonormal coordinates of the complement retain identity covariance.
        u = null_space(hw.T) if h.shape[1] else np.eye(n)
        aw = basis['operator']/problem.sigma[:,j,None]
        a = u.T@aw
        matrices.append(a); observations.append(u.T@(problem.y[:,j]/problem.sigma[:,j]))
        projections.append(u)
        survival.append(float(np.linalg.norm(a)/np.linalg.norm(aw)))
    return basis,h,matrices,observations,projections,survival


def _positive_fit(a,y,r,lam):
    augmented = np.vstack([a, np.sqrt(lam)*r, 1e-6*np.eye(a.shape[1])])
    rhs = np.r_[y, np.zeros(r.shape[0]+a.shape[1])]
    fit = lsq_linear(augmented,rhs,bounds=(0,np.inf),method="bvls",tol=1e-10,max_iter=1000)
    if not fit.success:
        raise RuntimeError("Positive spline least squares did not converge")
    return fit.x


def _choose_smoothness(a,y,r,config):
    if config.smoothness is not None:
        x = _positive_fit(a,y,r,config.smoothness)
        return config.smoothness,x,[dict(smoothness=config.smoothness,rss=float(np.sum((a@x-y)**2)))]
    target = config.discrepancy_ratio*len(y)
    path = []
    best = None
    # Fixed data-only Morozov path, never selected from a true distribution.
    for lam in np.r_[0.,np.logspace(-6,6,25)]:
        x = _positive_fit(a,y,r,float(lam)); rss=float(np.sum((a@x-y)**2))
        path.append(dict(smoothness=float(lam),rss=rss))
        if rss <= target or best is None:
            best=(float(lam),x)
        elif best[0] != 0. or len(path) > 1:
            break
    return *best,path


def _joint_admm(matrices, observations, r, lambdas, initial, groups, tau, config):
    """Two-block consensus ADMM: smooth x versus (positive z, group-SVT v)."""
    x=initial.copy(); z=x.copy(); v=x.copy(); uz=np.zeros_like(x); uv=uz.copy()
    sets=[np.flatnonzero(groups==g) for g in np.unique(groups)]
    q=[a.T@a+lam*(r.T@r)+1e-12*np.eye(x.shape[0]) for a,lam in zip(matrices,lambdas)]
    rhs=[a.T@y for a,y in zip(matrices,observations)]
    rho=1.; trace=[]; success=False
    def factors():return [cho_factor(t+2*rho*np.eye(x.shape[0])) for t in q]
    chol=factors()
    for iteration in range(1,config.max_iterations+1):
        for j in range(x.shape[1]):
            x[:,j]=cho_solve(chol[j],rhs[j]+rho*(z[:,j]-uz[:,j]+v[:,j]-uv[:,j]))
        oldz=z.copy(); oldv=v.copy()
        z=np.maximum(x+uz,0)
        v=x+uv
        for group in sets:
            if len(group)>1:
                u,s,vt=np.linalg.svd(v[:,group],full_matrices=False)
                v[:,group]=(u*np.maximum(s-tau/rho,0))@vt
        uz+=x-z; uv+=x-v
        primal=float(np.sqrt(np.sum((x-z)**2)+np.sum((x-v)**2)))
        dual=float(rho*np.linalg.norm((z-oldz)+(v-oldv)))
        ep=config.tolerance*(np.sqrt(2*x.size)+max(np.sqrt(2)*np.linalg.norm(x),np.sqrt(np.sum(z*z)+np.sum(v*v))))
        ed=config.tolerance*(np.sqrt(x.size)+rho*np.linalg.norm(uz+uv))
        if iteration==1 or iteration%25==0 or (primal<=ep and dual<=ed):
            trace.append(dict(iteration=iteration,primal=primal,dual=dual,primal_tolerance=float(ep),dual_tolerance=float(ed),rho=rho))
        if primal<=ep and dual<=ed:
            success=True;break
        if iteration%25==0:
            # Balance residuals relative to their different stopping scales.
            rp=primal/ep;rd=dual/ed
            newrho=rho*2 if rp>10*rd else rho/2 if rd>10*rp else rho
            newrho=float(np.clip(newrho,1e-6,1e6))
            if newrho!=rho:
                uz*=rho/newrho;uv*=rho/newrho;rho=newrho;chol=factors()
    return z,dict(converged=success,iterations=iteration,trace=trace)


def _solve(problem, config, joint, nuisance_profiles=None, nuisance_bounds=None, reference_source=None):
    basis,h,aa,yy,projections,survival=_prepare(problem,config,nuisance_profiles,nuisance_bounds,reference_source)
    groups,scales=compatible_groups(problem.y,problem.sigma,config.group_probability,h)
    if not joint:
        groups=np.arange(problem.y.shape[1]);scales=np.ones(problem.y.shape[1])
    # Fixed empirical amplitude normalization for DOSY; noise scaling remains exact conditional on scales.
    fits=[_choose_smoothness(a,y,basis['roughness'],config) for a,y in zip(aa,yy)]
    aa=[a*scale for a,scale in zip(aa,scales)]
    lambdas=[v[0]*scale**2 for v,scale in zip(fits,scales)]
    coeff=np.column_stack([v[1]/scale for v,scale in zip(fits,scales)])
    diag=dict(converged=True,iterations=0,trace=[])
    if joint and config.low_rank>0 and len(np.unique(groups))<len(groups):
        coeff,diag=_joint_admm(aa,yy,basis['roughness'],lambdas,coeff,groups,config.low_rank,config)
    coeff*=scales
    target=basis['operator']@coeff
    amplitudes=np.empty((h.shape[1],problem.y.shape[1]))
    cancelled=np.empty_like(problem.y)
    for j,u in enumerate(projections):
        amplitudes[:,j]=np.linalg.lstsq(h/problem.sigma[:,j,None],(problem.y[:,j]-target[:,j])/problem.sigma[:,j],rcond=None)[0]
        # The cancelled signal is a linear projection, distinct from fitted target.
        cancelled[:,j]=problem.sigma[:,j]*(u@(u.T@(problem.y[:,j]/problem.sigma[:,j])))
    reconstructed=target+h@amplitudes
    diag.update(method="RAI continuous DOSY experimental" if joint else "RAI continuous 1D experimental",
                representation="nonnegative cubic density per log D",smoothness=[v[0] for v in fits],
                selection_paths=[f[2] for f in fits],normalized_scales=scales.tolist(),
                nuisance_count=h.shape[1],reference_source=reference_source,
                nuisance_amplitudes_nonnegative=bool(np.all(amplitudes>=0)),
                target_operator_survival=survival,projected_noise_dof=len(problem.b)-h.shape[1],
                reduced_chi_square=float(np.sum(((reconstructed-problem.y)/problem.sigma)**2)/((len(problem.b)-h.shape[1])*problem.y.shape[1])),
                reference_uncertainty_propagated=False,groups_are_heuristic=True,
                scientific_status="experimental; no established novelty or superiority")
    return ContinuousResult(basis['edges'],basis['rates'],basis['mass_basis']@coeff,
                            basis['density_basis']@coeff,coeff,reconstructed,problem.y-reconstructed,
                            cancelled,amplitudes,groups,diag,asdict(config))


def solve_single_profile(problem: DecayProblem, config=ContinuousConfig(), **references):
    """Recover a continuous distribution from ONE attenuation curve, with optional reference cancellation."""
    if problem.y.shape[1]!=1:
        raise ValueError("Use solve_dosy_profiles for multiple spectral columns")
    return _solve(problem,config,False,**references)


def solve_dosy_profiles(problem: DecayProblem, config=ContinuousConfig(), **references):
    """Recover continuous distributions jointly using nonlocal spectral-column groups."""
    return _solve(problem,config,True,**references)
