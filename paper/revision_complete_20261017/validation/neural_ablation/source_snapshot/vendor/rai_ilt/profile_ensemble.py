"""Experimental extension: wider model-order search and ambiguity averaging.

These weights are sensitivity summaries, NOT Bayesian model probabilities.
No training data, true component count, or experimental modifications enter.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr,log_ndtr
from .mixed_unmixing import MixedConfig,ProfileOperator,solve_mixed_single,solve_mixed_dosy
from .model import DecayProblem

@dataclass(frozen=True)
class EnsembleConfig:
    max_factors:int=6
    starts:int=4
    averaging:str='aic'
    def __post_init__(self):
        if self.averaging not in ('selected','adequate_uniform','aic'):raise ValueError('Invalid averaging rule')
        MixedConfig(max_factors=self.max_factors,starts=self.starts)

@dataclass
class EnsembleResult:
    rate_edges:np.ndarray
    continuous_masses:np.ndarray
    atomic_masses:np.ndarray
    reconstructed:np.ndarray
    diagnostics:dict
    candidates:list
    weights:np.ndarray
    @property
    def masses(self):return self.continuous_masses+self.atomic_masses

def candidate_arrays(candidate,b,edges):
    rank=candidate['rank'];broad=candidate['broad'];bins=len(edges)-1
    if rank==0:raise ValueError('Rank-zero models must use the base result directly')
    locations=np.asarray(candidate['locations_unsorted']);widths=np.asarray(candidate['log_widths_unsorted']);skew=np.asarray(candidate['skewness_unsorted'])
    spectra=np.asarray(candidate['spectra_unsorted']);mass=np.zeros((bins,rank));z=np.log(edges)
    q,w=np.polynomial.legendre.leggauss(12);zz=(z[:-1,None]+z[1:,None])/2+np.diff(z)[:,None]*q/2
    for k in range(rank):
        if k<broad:
            u=(zz-np.log(locations[k]))/widths[k];logp=-.5*u*u+log_ndtr(skew[k]*u)
            mass[:,k]=np.sum(np.exp(logp)*w,axis=1)*np.diff(z)/2;mass[:,k]/=mass[:,k].sum()
        else:mass[np.clip(np.searchsorted(edges,locations[k])-1,0,bins-1),k]=1
    op=ProfileOperator(b,(edges[0],edges[-1]),256,True)
    theta=np.r_[np.log(locations)-op.origin,np.log(widths[:broad]),skew[:broad]]
    h,_,_=op.evaluate(theta,rank,broad)
    return mass[:,:broad]@spectra[:broad],mass[:,broad:]@spectra[broad:],h@spectra

def solve_profile_ensemble(problem:DecayProblem,config=EnsembleConfig()):
    base=(solve_mixed_single if problem.y.shape[1]==1 else solve_mixed_dosy)(problem,
        MixedConfig(max_factors=config.max_factors,starts=config.starts))
    candidates=base.candidates;selected=next(i for i,c in enumerate(candidates) if c['selected'])
    rank=candidates[selected]['rank']
    eligible=[i for i,c in enumerate(candidates) if c['rank']==rank and c['converged'] and c['noise_adequate']]
    if config.averaging=='selected' or not eligible or rank==0:eligible=[selected]
    weights=np.zeros(len(candidates))
    if config.averaging=='aic' and len(eligible)>1:
        score=np.array([candidates[i]['rss']+2*candidates[i]['parameters'] for i in eligible]);w=np.exp(-.5*(score-score.min()));w/=w.sum()
    else:w=np.ones(len(eligible))/len(eligible)
    weights[eligible]=w
    if len(eligible)==1:
        continuous=base.continuous_masses.copy();atoms=base.atomic_masses.copy();pred=base.reconstructed.copy()
    else:
        continuous=np.zeros_like(base.masses);atoms=np.zeros_like(base.masses);pred=np.zeros_like(problem.y)
        for i in eligible:
            c,a,h=candidate_arrays(candidates[i],problem.b,base.rate_edges)
            continuous+=weights[i]*c;atoms+=weights[i]*a;pred+=weights[i]*h
    diagnostics=dict(base.diagnostics)
    diagnostics.update(method='RAI profile ensemble experimental',max_factors=config.max_factors,
        averaged_type_models=len(eligible),averaging=config.averaging,weights=weights.tolist(),
        posterior_probabilities=False,measure_type_resolved=len(eligible)==1,
        factor_identity_for_ensemble=False,ensemble_noise_rmse=float(np.sqrt(np.mean(((pred-problem.y)/problem.sigma)**2))))
    return EnsembleResult(base.rate_edges,continuous,atoms,pred,diagnostics,candidates,weights)
