"""RAI moment inversion: positive measures, shared structure, explicit diagnostics."""
from dataclasses import dataclass, field, asdict
import time
import numpy as np
from scipy.optimize import nnls, minimize
from .model import DecayProblem, real_array
from .solver import RAIResult
from .components import exponential_kernel, _profile
from .moment_space import MomentRegion, MomentSpace, adaptive_space, radau_measure, moment_sequence, hankel
from .moment_solver import MomentConfig, fit_moments


class MomentConvergenceError(RuntimeError):
    """No accepted feasible initial iterate; diagnostics remain accessible."""
    def __init__(self,diagnostics,history):
        super().__init__("Moment conic initialization failed; no reliable measure can be returned")
        self.diagnostics=diagnostics
        self.history=history


@dataclass
class MomentResult(RAIResult):
    moments: np.ndarray | None = None
    moment_regions: list = field(default_factory=list)
    quadrature_rates: np.ndarray | None = None
    quadrature_amplitudes: np.ndarray | None = None
    seed_moments: np.ndarray | None = None
    seed_quadrature_rates: np.ndarray | None = None
    seed_quadrature_amplitudes: np.ndarray | None = None

    def predict(self,b):
        b=real_array(b,"b").reshape(-1)
        if np.any(b<0):
            raise ValueError("b must be nonnegative")
        return exponential_kernel(b,self.quadrature_rates)@self.quadrature_amplitudes


def _selection_rows(b,stride):
    unique=np.unique(b)
    if len(unique)<8:
        raise ValueError("Moment selection needs at least eight distinct acquisition values")
    selected=unique[2:-1:stride]
    # Replicate measurements of the same acquisition stay in the same fold.
    validation=np.isin(b,selected)
    return ~validation,validation


def _normalization(problem,train):
    y,sigma,b=problem.y[train],problem.sigma[train],problem.b[train]
    scales=np.maximum(np.max(np.abs(y),axis=0),8*np.median(sigma,axis=0))
    minimum=b==b.min()
    weights=1/sigma[minimum]**2
    estimate=np.sum(y[minimum]*weights,axis=0)/weights.sum(axis=0)
    se=1/np.sqrt(weights.sum(axis=0))
    exponent=b.min()*problem.rate_bounds[1]
    if exponent>30:
        raise ValueError("Earliest acquisition does not constrain total mass within numerical range")
    # This is a declared data-dependent eight-sigma cap, not a certainty.
    caps=(np.maximum(estimate,0)+8*se)*np.exp(exponent)/scales
    return scales,caps


def _positive_quadrature(column,degree,config):
    mass=float(max(column[0],0))
    if mass<=10*config.cone_tolerance:
        nodes=np.array([0.]) if mass>0 else np.empty(0)
        weights=np.array([mass]) if mass>0 else np.empty(0)
        mismatch=float(np.max(np.abs(moment_sequence(nodes,weights,degree)[:,0]-column)))
        return nodes,weights,mismatch,"negligible_mass_center"
    try:
        nodes,weights,error=radau_measure(column,tolerance=config.extraction_tolerance)
        if error<=max(50*config.cone_tolerance,config.extraction_tolerance*mass):
            return nodes,weights,error,"radau"
    except ValueError:
        pass
    # Numerical integration fallback only; these nodes are never fitted bins
    # in the inversion and are not declared resolved components.
    nodes=np.cos(np.linspace(np.pi,0,257))
    powers=nodes[None,:]**np.arange(degree+1)[:,None]
    weights=nnls(powers,column,maxiter=10000)[0]
    if weights.sum()>0:
        weights*=mass/weights.sum()
    else:
        weights[len(nodes)//2]=mass
    use=weights>0
    error=float(np.max(np.abs(powers@weights-column)))
    return nodes[use],weights[use],error,"nonnegative_quadrature_fallback"


def quadrature_representation(space,m,config):
    rates=[]; amplitudes=[]; report=[]
    p=m.shape[1]
    for c,(sl,r) in enumerate(zip(space.slices(),space.regions)):
        for q in range(p):
            u,w,error,method=_positive_quadrature(m[sl,q],r.degree,config)
            for value,mass in zip(u,w):
                row=np.zeros(p); row[q]=mass
                rates.append(r.center+r.halfwidth*value); amplitudes.append(row)
            report.append(dict(region=c,signal=q,moment_mismatch=error,method=method,nodes=len(u)))
    if not rates:
        return np.empty(0),np.zeros((0,p)),report
    order=np.argsort(rates,kind="stable")
    return np.array(rates)[order],np.array(amplitudes)[order],report


def _rank_summary(space,m,config):
    summaries=[]
    for i,(sl,r) in enumerate(zip(space.slices(),space.regions)):
        pooled=m[sl].mean(axis=1)
        eigen=np.linalg.eigvalsh(hankel(pooled,r.degree//2+1))
        scale=max(eigen[-1],1e-12)
        rank=int(np.sum(eigen>max(config.extraction_tolerance*scale,10*config.cone_tolerance)))
        previous=np.linalg.eigvalsh(hankel(pooled,r.degree//2))
        smaller=int(np.sum(previous>max(config.extraction_tolerance*scale,10*config.cone_tolerance)))
        summaries.append(dict(region=i,mass=float(pooled[0]),rank=rank,previous_rank=smaller,
                              numerically_flat=rank==smaller and rank>0,
                              eigenvalues=eigen.tolist()))
    return summaries


def _component_candidate(space,m,b,y,sigma,config):
    """Candidate common atoms extracted from pooled moments, followed by NNLS.

    This is a separate approximation whose discrepancy from the full moment
    reconstruction is retained. It is never used silently for prediction.
    """
    nodes=[]
    for sl,r in zip(space.slices(),space.regions):
        pooled=m[sl].mean(axis=1)
        if pooled[0]<=max(config.mass_threshold*m[space.mass_indices].mean(axis=1).sum(),10*config.cone_tolerance):
            continue
        u,w,_,_=_positive_quadrature(pooled,r.degree,config)
        selected=w>config.mass_threshold*max(pooled[0],1e-12)
        nodes.extend(r.center+r.halfwidth*u[selected])
    if not nodes:
        return np.empty(0),np.zeros((0,y.shape[1])),dict(polished=False)
    nodes=np.sort(np.array(nodes))
    # Remove numerical duplicates only, not nearby physical components.
    nodes=nodes[np.r_[True,np.diff(np.log(nodes))>1e-6]]
    if len(nodes)>config.max_components:
        return np.empty(0),np.zeros((0,y.shape[1])),dict(polished=False,too_many_candidates=True)
    def amplitudes(rates):
        k=exponential_kernel(b,rates)
        return np.column_stack([nnls(k/sigma[:,q,None],y[:,q]/sigma[:,q],maxiter=1000)[0] for q in range(y.shape[1])])
    a=amplitudes(nodes)
    active=np.max(a,axis=1)>config.mass_threshold*max(float(np.max(a.sum(axis=0))),1e-12)
    nodes=nodes[active]
    a=amplitudes(nodes) if len(nodes) else np.zeros((0,y.shape[1]))
    diagnostics=dict(polished=False)
    if config.polish and len(nodes):
        reference=float(np.exp(np.mean(np.log([space.regions[0].lower,space.regions[-1].upper]))))
        lower,upper=np.log(np.array([space.regions[0].lower,space.regions[-1].upper])/reference)
        def objective(z):
            value,gradient,_=_profile(z,b*reference,y,sigma)
            return value,gradient
        opt=minimize(objective,np.log(nodes/reference),jac=True,method="L-BFGS-B",bounds=[(lower,upper)]*len(nodes),
                     options=dict(maxiter=500,ftol=1e-12,gtol=1e-6,maxls=40))
        diagnostics.update(polished=bool(opt.success),polish_status=str(opt.message))
        if opt.success:
            nodes=np.sort(np.exp(opt.x)*reference); a=amplitudes(nodes)
    return nodes,a,diagnostics


def _propose_refinement(space,m,config):
    ranks=_rank_summary(space,m,config)
    eligible=[r for r in ranks if r['rank']>=2 and r['mass']>config.mass_threshold]
    if not eligible:
        return None,None
    chosen=max(eligible,key=lambda r:r['mass']*r['rank'])['region']
    region=space.regions[chosen]
    new=list(space.regions)
    if region.degree+2<=config.max_degree:
        new[chosen]=MomentRegion(region.lower,region.upper,region.degree+2)
        action=dict(operation="increase_order_for_active_structure",region=chosen)
    elif len(new)<config.max_regions:
        mid=float(np.sqrt(region.lower)*np.sqrt(region.upper))
        new[chosen:chosen+1]=[MomentRegion(region.lower,mid,region.degree),MomentRegion(mid,region.upper,region.degree)]
        action=dict(operation="split_active_moment_region",region=chosen)
    else:
        return None,None
    return MomentSpace(new),action


def solve_moments(problem,config=None,*,prior=None):
    """Fit RAI-ILT/RAI-Net moments without knowing the component count.

    Continuous recovery selects a finite positive model, then recomputes all
    output representations from it while retaining the original moment seed.
    In quadrature mode, component candidates remain separate diagnostics.
    """
    if not isinstance(problem,DecayProblem):
        raise TypeError("problem must be a DecayProblem")
    config=config or MomentConfig()
    if not isinstance(config,MomentConfig):
        raise TypeError("config must be MomentConfig")
    start=time.perf_counter()
    train,validation=_selection_rows(problem.b,config.validation_stride)
    scales,caps=_normalization(problem,train)
    y,sigma=problem.y/scales,problem.sigma/scales
    space,adaptation=adaptive_space(problem.b[train],problem.rate_bounds,sigma[train],caps,config)
    history=[]; candidates=[]
    for penalty in config.penalties:
        current=space
        for refinement in range(config.max_refinements+1):
            reference=None if prior is None else prior.reference(problem.b[train],y[train],sigma[train],current)
            strength=config.prior_strength if prior is not None else 0.
            m,trace,diag=fit_moments(current,problem.b[train],y[train],sigma[train],caps,config,penalty,reference,strength)
            if diag['reason']=='initial_cone_failure':
                raise MomentConvergenceError(diag,trace)
            rates,a,quad=quadrature_representation(current,m,config)
            pred=exponential_kernel(problem.b[validation],rates)@a
            losses=np.mean(((pred-y[validation])/sigma[validation])**2,axis=1)
            rank=_rank_summary(current,m,config)
            item=dict(penalty=penalty,refinement=refinement,regions=current.metadata(),optimizer=diag,
                      validation_loss=float(losses.mean()),validation_row_losses=losses.tolist(),
                      rank_summary=rank,effective_rank=sum(r['rank'] for r in rank),trace=trace,
                      quadrature=quad,accepted=False)
            history.append(item)
            candidates.append((current,m,rates,a,item))
            if refinement>=config.max_refinements or not diag['success']:
                break
            current,action=_propose_refinement(current,m,config)
            if current is None:
                break
            adaptation.append(dict(penalty=penalty,**action))
    valid=[c for c in candidates if c[4]['optimizer']['success'] and np.isfinite(c[4]['validation_loss'])]
    pool=valid or candidates
    selected=min(pool,key=lambda c:c[4]['validation_loss'])
    if config.selection_rule=="one_se":
        best=np.array(selected[4]['validation_row_losses'])
        se=float(np.std(best,ddof=1)/np.sqrt(len(best))) if len(best)>1 else 0.
        eligible=[c for c in pool if c[4]['validation_loss']<=best.mean()+se]
        selected=min(eligible,key=lambda c:(c[4]['effective_rank'],c[0].size,c[4]['validation_loss']))
    current,_,_,_,choice=selected
    choice['accepted']=True
    reference=None if prior is None else prior.reference(problem.b,y,sigma,current)
    m,trace,diag=fit_moments(current,problem.b,y,sigma,caps,config,choice['penalty'],reference,
                            config.prior_strength if prior is not None else 0.)
    if diag['reason']=='initial_cone_failure':
        raise MomentConvergenceError(diag,trace)
    rates,a,quad=quadrature_representation(current,m,config)
    pred=exponential_kernel(problem.b,rates)@a
    polynomial=current.kernel(problem.b)@m
    quadrature_difference=float(np.max(np.abs(pred-polynomial)/sigma))
    bound=current.bounds(problem.b)@np.maximum(m[current.mass_indices],0)
    kernel_bound=float(np.max(bound/sigma))
    components,component_a,component_diag=_component_candidate(current,m,problem.b,y,sigma,config)
    component_pred=exponential_kernel(problem.b,components)@component_a
    component_difference=float(np.sqrt(np.mean(((component_pred-pred)/sigma)**2)))
    ranks=_rank_summary(current,m,config)
    flat=all(r['numerically_flat'] for r in ranks if r['mass']>config.mass_threshold)
    # Empirical label only. It is not a calibrated resolution certificate.
    state=("discrete_candidate" if len(components) and flat and component_difference<=.2 else "unresolved_moment_measure")
    if not diag['success'] or not valid:
        state="incomplete_optimization"
    approximation_ok=quadrature_difference<=max(.1,4*config.model_error_budget) and kernel_bound<=config.model_error_budget*1.01
    success=bool(diag['success'] and valid and approximation_ok)
    if not approximation_ok:
        state="approximation_check_failed"
    edges=np.geomspace(*problem.rate_bounds,config.display_bins+1)
    histogram=np.zeros((config.display_bins,y.shape[1]))
    positions=np.clip(np.searchsorted(edges,rates,side="right")-1,0,config.display_bins-1)
    for j,position in enumerate(positions): histogram[position]+=a[j]*scales
    diagnostics=dict(method="RAI-Net moments" if prior is not None else "RAI-ILT moments",state=state,
                     optimizer={**diag,"success":success,"algorithm":"positive moment semidefinite majorization"},
                     numerical_success_is_not_resolution=True,selected_penalty=choice['penalty'],
                     selection_criterion=f"held-out acquisition risk ({config.selection_rule})",
                     validation_loss=choice['validation_loss'],selection_rows_original=np.flatnonzero(validation).tolist(),
                     selection_candidates=len(candidates),selection_candidates_converged=len(valid),
                     rank_summary=ranks,adaptation=adaptation,final_trace=trace,quadrature=quad,
                     kernel_error_bound_in_sigma=kernel_bound,quadrature_discrepancy_in_sigma=quadrature_difference,
                     approximation_verified=bool(approximation_ok),component_candidate_rates=components.tolist(),
                     component_candidate_amplitudes=(component_a*scales).tolist(),
                     component_candidate_discrepancy_rms_sigma=component_difference,component_extraction=component_diag,
                     learned_prior=prior is not None,prior_metadata=None if prior is None else prior.metadata,
                     scales=scales.tolist(),mass_caps=(caps*scales).tolist(),
                     noise_model="independent Gaussian with supplied sigma; correlated covariance unsupported",
                     display_kind="positive quadrature mass histogram, not an independent density estimate",
                     uncertainty_interpretation="numerical rank and predictive diagnostics, not calibrated confidence",
                     weighted_rmse=float(np.sqrt(np.mean(((pred-y)/sigma)**2))),seconds=time.perf_counter()-start)
    settings=asdict(config); settings['representation']='moments'
    result=MomentResult(rate_edges=edges,masses=histogram,reconstructed=pred*scales,residual=(y-pred)*scales,
                        graph=np.zeros((y.shape[1],y.shape[1])),history=history,diagnostics=diagnostics,config=settings,
                        moments=m*scales,moment_regions=current.metadata(),quadrature_rates=rates,
                        quadrature_amplitudes=a*scales)
    if config.recovery=="continuous":
        _recover_continuous(problem,result,config)
    result.diagnostics['seconds']=time.perf_counter()-start
    return result


def _recover_continuous(problem,result,config):
    """Replace the integration representation by an explicitly selected model.

    Retain the original moment seed and its independent convergence flags.
    Recompute every output array from the same final positive measure.
    """
    from .support import recover_support
    d,a,history,diagnostics=recover_support(problem.b,problem.y,problem.sigma,problem.rate_bounds,
        result.quadrature_rates,result.quadrature_amplitudes,
        max_components=config.support_limit,max_iterations=config.support_iterations)
    result.seed_moments=result.moments.copy()
    result.seed_quadrature_rates=result.quadrature_rates.copy()
    result.seed_quadrature_amplitudes=result.quadrature_amplitudes.copy()
    seed_optimizer=result.diagnostics['optimizer'].copy()
    space=MomentSpace([MomentRegion(**r) for r in result.moment_regions])
    result.component_rates=d;result.component_amplitudes=a
    result.quadrature_rates=d;result.quadrature_amplitudes=a
    result.moments=space.from_measure(d,a)
    result.reconstructed=exponential_kernel(problem.b,d)@a
    result.residual=problem.y-result.reconstructed
    result.masses[:]=0
    locations=np.clip(np.searchsorted(result.rate_edges,d,side='right')-1,0,len(result.masses)-1)
    for i,location in enumerate(locations):result.masses[location]+=a[i]
    bound=space.bounds(problem.b)@result.moments[space.mass_indices]
    kernel_bound=float(np.max(bound/problem.sigma))
    mismatch=float(np.max(abs(space.kernel(problem.b)@result.moments-result.reconstructed)/problem.sigma))
    approximation=kernel_bound<=config.model_error_budget*1.01 and mismatch<=max(.1,4*config.model_error_budget)
    success=diagnostics['optimizer_success'] and diagnostics['noise_fit_adequate'] and approximation
    precise=diagnostics['precision_supported'] and not diagnostics['model_order_ambiguous'] and not diagnostics['rate_bound_hit']
    state=('atomic_model_locally_precise' if precise else 'atomic_model_uncertain') if len(d) else 'zero_signal_candidate'
    if not success:state='incomplete_or_inadequate_support_fit'
    result.diagnostics.update(state=state,optimizer=dict(success=bool(success),algorithm=diagnostics['algorithm'],
        objective=.5*diagnostics['selected_rss'],global_optimum_certified=False),
        moment_seed_optimizer=seed_optimizer,moment_seed_selection_criterion=result.diagnostics['selection_criterion'],
        selection_criterion=diagnostics['selection_criterion'],selection_uses_all_rows=True,
        support_search=history,support_diagnostics=diagnostics,component_candidate_rates=d.tolist(),
        component_candidate_amplitudes=a.tolist(),component_candidate_discrepancy_rms_sigma=0.,
        component_extraction=dict(method='birth-slide-delete',prediction_uses_selected_components=True),
        seed_kernel_error_bound_in_sigma=result.diagnostics['kernel_error_bound_in_sigma'],
        seed_quadrature_discrepancy_in_sigma=result.diagnostics['quadrature_discrepancy_in_sigma'],
        kernel_error_bound_in_sigma=kernel_bound,quadrature_discrepancy_in_sigma=mismatch,
        approximation_verified=bool(approximation),mass_caps_scope='initial moment seed only',
        rank_summary_scope='initial moment seed only',
        display_kind='selected continuous-rate model mass histogram; finite-mixture assumption',
        uncertainty_interpretation=diagnostics['uncertainty_interpretation'],
        weighted_rmse=float(np.sqrt(np.mean((result.residual/problem.sigma)**2))))


def solve_moment_net(problem,prior,config=None):
    from .moment_net import MomentPrior
    if not isinstance(prior,MomentPrior):
        raise TypeError("prior must be a trained MomentPrior")
    return solve_moments(problem,config,prior=prior)
