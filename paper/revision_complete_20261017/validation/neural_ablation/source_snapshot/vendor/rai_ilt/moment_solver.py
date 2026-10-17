"""Semidefinite majorization for positive local moments.

The backend exposes primal/dual residuals and gaps. No row-action projection
or fitted density bins are used in this solver.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy import sparse
from .moment_space import hankel, hankel_adjoint


@dataclass(frozen=True)
class MomentConfig:
    initial_regions: int = 4
    max_regions: int = 12
    min_degree: int = 4
    max_degree: int = 12
    model_error_budget: float = .03
    penalties: tuple = (0., .001, .01, .1)
    epsilon: float = .0001
    ridge: float = 1e-7
    joint: bool = True
    max_mm_iterations: int = 120
    mm_tolerance: float = 2e-5
    cone_tolerance: float = 2e-7
    max_cone_iterations: int = 250
    validation_stride: int = 4
    selection_rule: str = "minimum"
    display_bins: int = 256
    extraction_tolerance: float = 1e-5
    mass_threshold: float = .005
    max_components: int = 8
    polish: bool = False
    max_refinements: int = 1
    prior_strength: float = 1.
    recovery: str = "continuous"
    support_limit: int = 6
    support_iterations: int = 800

    def __post_init__(self):
        for name in ("initial_regions","max_regions","min_degree","max_degree","max_mm_iterations",
                     "max_cone_iterations","validation_stride","display_bins","max_components","max_refinements",
                     "support_limit","support_iterations"):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,np.integer)):
                raise ValueError(f"{name} must be an integer")
        if not 1<=self.initial_regions<=self.max_regions or self.min_degree<2 or self.max_degree<self.min_degree:
            raise ValueError("Invalid moment representation limits")
        if self.min_degree%2 or self.max_degree%2 or self.max_degree>24:
            raise ValueError("Moment degrees must be even and at most 24")
        if min(self.max_mm_iterations,self.max_cone_iterations,self.max_components)<1 or self.max_refinements<0:
            raise ValueError("Invalid iteration/search limit")
        if self.validation_stride<3 or self.display_bins<128:
            raise ValueError("Need validation stride >=3 and at least 128 display bins")
        for name in ("model_error_budget","epsilon","ridge","mm_tolerance","cone_tolerance","extraction_tolerance"):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0<=self.mass_threshold<1 or not np.isfinite(self.prior_strength) or self.prior_strength<0:
            raise ValueError("Invalid mass threshold or prior strength")
        values=np.asarray(self.penalties,dtype=float)
        if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or np.any(values<0):
            raise ValueError("Need a finite nonnegative penalty path")
        object.__setattr__(self,"penalties",tuple(float(x) for x in values))
        if self.selection_rule not in ("minimum","one_se"):
            raise ValueError("selection_rule must be minimum or one_se")
        if not isinstance(self.joint,bool) or not isinstance(self.polish,bool):
            raise ValueError("joint and polish must be booleans")
        if self.recovery not in ("continuous","quadrature") or self.support_limit<1 or self.support_iterations<1:
            raise ValueError("Invalid continuous support recovery configuration")


def _clarabel():
    try:
        import clarabel
    except ImportError as exc:
        raise ImportError("Moment inversion requires rai-ilt[moments]") from exc
    return clarabel


class MomentQuadratic:
    """Fixed cone and least-squares matrices; majorization changes linear costs."""
    def __init__(self,space,b,y,sigma,caps,config,reference=None,strength=0.):
        clarabel=_clarabel()
        self.space,self.y,self.sigma,self.caps,self.config=space,y,sigma,caps,config
        self.kernel=space.kernel(b)
        self.shape=(space.size,y.shape[1])
        self.reference=np.zeros(self.shape) if reference is None else np.array(reference,dtype=float)
        if self.reference.shape!=self.shape or not np.isfinite(self.reference).all():
            raise ValueError("Invalid learned reference moments")
        self.strength=float(strength)
        self.prior_weights=np.concatenate([np.full(r.degree+1,1/(r.degree+1)) for r in space.regions])[:,None]/y.shape[1]
        d=space.size
        penalty_diagonal=config.ridge+strength*self.prior_weights[:,0]
        matrices=[]
        for q in range(y.shape[1]):
            a=self.kernel/sigma[:,q,None]/np.sqrt(y.size)
            matrices.append(sparse.csc_matrix(a))
        self.moment_variables=d*y.shape[1]
        # Explicit residual variables avoid forming ill-conditioned normal
        # equations and cancellation of a large constant in the duality gap.
        self.P=sparse.diags(np.r_[np.tile(penalty_diagonal,y.shape[1]),np.ones(y.size)],format="csc")
        self.base_q=np.r_[(-strength*self.prior_weights*self.reference).ravel(order="F"),np.zeros(y.size)]
        cone_map,sizes=space.cone_map()
        mass=sparse.csc_matrix((np.ones(len(space.mass_indices)),(np.zeros(len(space.mass_indices),int),space.mass_indices)),shape=(1,d))
        one=sparse.vstack((-cone_map,mass),format="csc")
        cone_a=sparse.block_diag([one]*y.shape[1],format="csc")
        residual_a=sparse.hstack((sparse.block_diag(matrices,format="csc"),-sparse.eye(y.size)),format="csc")
        self.A=sparse.vstack((residual_a,sparse.hstack((cone_a,sparse.csc_matrix((cone_a.shape[0],y.size))))),format="csc")
        self.rhs=np.r_[((y/sigma)/np.sqrt(y.size)).ravel(order="F"),
                       np.concatenate([np.r_[np.zeros(cone_map.shape[0]),cap] for cap in caps])]
        cones=[clarabel.ZeroConeT(y.size)]
        for _ in range(y.shape[1]):
            cones.extend([clarabel.PSDTriangleConeT(s) for s in sizes])
            cones.append(clarabel.NonnegativeConeT(1))
        settings=clarabel.DefaultSettings()
        settings.verbose=False
        settings.max_iter=config.max_cone_iterations
        settings.tol_gap_abs=config.cone_tolerance*.5
        settings.tol_gap_rel=min(1e-10,config.cone_tolerance*.01)
        settings.tol_feas=min(1e-9,config.cone_tolerance*.01)
        settings.max_threads=1
        self.cones,self.settings=cones,settings
        # Public data updates preserve matrix sparsity for each MM iteration.
        self.backend=clarabel.DefaultSolver(self.P,self.base_q,self.A,self.rhs,cones,settings)

    def smooth_value(self,m):
        residual=(self.kernel@m-self.y)/self.sigma
        return (.5*float(np.mean(residual**2))+.5*self.config.ridge*float(np.sum(m*m))+
                .5*self.strength*float(np.sum(self.prior_weights*(m-self.reference)**2)))

    def _attempt(self,cost):
        self.backend.update(q=cost)
        solution=self.backend.solve()
        info=self.backend.get_info()
        m=np.array(solution.x[:self.moment_variables]).reshape(self.shape,order="F")
        feasible=self.space.feasibility(m,self.caps) if np.isfinite(m).all() else dict(min_cone_eigenvalue=-1e300,mass_cap_violation=1e300)
        tol=self.config.cone_tolerance
        ok=(str(solution.status) in ("Solved","AlmostSolved") and np.isfinite(m).all() and
            feasible['min_cone_eigenvalue']>=-tol and feasible['mass_cap_violation']<=tol and
            float(info.gap_abs)<=tol*max(1.,self.smooth_value(m)) and
            float(info.res_primal)<=tol and float(info.res_dual)<=tol)
        diagnostics=dict(success=bool(ok),status=str(solution.status),iterations=int(solution.iterations),
                         acceptance="explicit gap, primal/dual residual and moment-cone checks",
                         gap_abs=float(info.gap_abs),gap_rel=float(info.gap_rel),
                         primal_residual=float(info.res_primal),dual_residual=float(info.res_dual),
                         seconds=float(solution.solve_time),**feasible)
        return m,diagnostics

    def solve(self,linear):
        cost=self.base_q+np.r_[np.asarray(linear).ravel(order="F"),np.zeros(self.y.size)]
        m,diagnostics=self._attempt(cost)
        if not diagnostics['success']:
            # A bounded numerical retry changes factorization regularization,
            # never the objective, cones, or the acceptance tolerances.
            first=dict(diagnostics)
            settings=self.settings
            settings.static_regularization_constant=1e-10
            settings.max_step_fraction=.9
            settings.iterative_refinement_max_iter=20
            self.backend=_clarabel().DefaultSolver(self.P,cost,self.A,self.rhs,self.cones,settings)
            retry,retry_info=self._attempt(cost)
            m,diagnostics=retry,retry_info
            diagnostics['retry_after']=first
        return m,diagnostics


def rank_energy(space,m,epsilon,joint=True):
    """Average shared/independent log-det energy and its exact moment gradient."""
    energy=0.; gradient=np.zeros_like(m)
    p=m.shape[1]
    for sl,r in zip(space.slices(),space.regions):
        columns=[m[sl].mean(axis=1)] if joint else list(m[sl].T)
        for q,column in enumerate(columns):
            h=hankel(column,r.degree//2+1)+epsilon*np.eye(r.degree//2+1)
            eigen=np.linalg.eigvalsh(h)
            if eigen[0]<=0:
                raise ValueError("Log-det outside its positive-definite domain")
            factor=1. if joint else 1/p
            energy+=factor*float(np.log(eigen).sum())
            g=hankel_adjoint(np.linalg.inv(h),r.degree)
            if joint:
                gradient[sl]+=g[:,None]/p
            else:
                gradient[sl,q]+=g/p
    return energy,gradient


def fit_moments(space,b,y,sigma,caps,config,penalty,reference=None,strength=0.):
    start=time.perf_counter()
    qp=MomentQuadratic(space,b,y,sigma,caps,config,reference,strength)
    m,first=qp.solve(np.zeros((space.size,y.shape[1])))
    history=[dict(iteration=0,accepted=first['success'],subproblem=first)]
    if not first['success']:
        return m,history,dict(success=False,reason="initial_cone_failure",seconds=time.perf_counter()-start)
    previous=qp.smooth_value(m)+penalty*rank_energy(space,m,config.epsilon,config.joint)[0]
    history[0]['objective']=previous
    converged=penalty==0
    reason="unpenalized_optimum" if converged else "iteration_limit"
    for iteration in range(1,1 if converged else config.max_mm_iterations+1):
        _,gradient=rank_energy(space,m,config.epsilon,config.joint)
        candidate,diagnostics=qp.solve(penalty*gradient)
        record=dict(iteration=iteration,accepted=False,subproblem=diagnostics)
        history.append(record)
        if not diagnostics['success']:
            reason="cone_failure"; break
        objective=qp.smooth_value(candidate)+penalty*rank_energy(space,candidate,config.epsilon,config.joint)[0]
        gain=previous-objective
        step=float(np.linalg.norm(candidate-m)/max(1.,np.linalg.norm(m)))
        record.update(objective=objective,gain=gain,relative_step=step)
        if gain < -config.cone_tolerance*max(1.,abs(previous)):
            reason="descent_check_failed"; break
        record['accepted']=True
        m=candidate; previous=objective
        if abs(gain)<=config.mm_tolerance*max(1.,abs(objective)) and step<=np.sqrt(config.mm_tolerance):
            converged=True; reason="majorization_converged"; break
    return m,history,dict(success=bool(converged),reason=reason,objective=float(previous),
                         smooth_objective=qp.smooth_value(m),iterations=len(history)-1,
                         last_accepted_cone_success=True,seconds=time.perf_counter()-start,
                         **space.feasibility(m,caps))
