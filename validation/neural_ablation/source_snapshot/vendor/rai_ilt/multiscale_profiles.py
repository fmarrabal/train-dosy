"""Experimental multiscale inversion and shared/private DOSY reconstruction.

Nonnegative bin masses are estimated without a prescribed density family or
component count. Adaptation and data-derived anchors are heuristic; conditional
quadratic subproblems are convex. Neither novelty nor chemical identifiability
is asserted. No acquisition changes or external cancellation references enter.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy import sparse
from scipy.optimize import nnls
from scipy.linalg import cholesky,solve_triangular
from functools import lru_cache
from scipy.ndimage import maximum_filter1d
from scipy.stats import chi2
import clarabel
from .model import DecayProblem

@dataclass(frozen=True)
class MultiscaleConfig:
    bins: int = 256
    discrepancy_slack: float = .1
    adaptation: float = .5
    sharing: float = 1.
    max_anchors: int = 8
    path_steps: int = 10
    tolerance: float = 1e-8
    constrain_sharing: bool = True
    def __post_init__(self):
        if not isinstance(self.bins,int) or self.bins<32:raise ValueError('bins must be >=32')
        if not isinstance(self.max_anchors,int) or self.max_anchors<1:raise ValueError('max_anchors must be positive')
        if not isinstance(self.path_steps,int) or self.path_steps<4:raise ValueError('path_steps must be >=4')
        for key in ('discrepancy_slack','adaptation','sharing'):
            if not np.isfinite(getattr(self,key)) or getattr(self,key)<0:raise ValueError(key+' must be finite and nonnegative')
        if not np.isfinite(self.tolerance) or self.tolerance<=0:raise ValueError('tolerance must be positive')

@dataclass
class MultiscaleResult:
    rate_edges: np.ndarray
    masses: np.ndarray
    reconstructed: np.ndarray
    independent_masses: np.ndarray
    shared_masses: np.ndarray
    private_masses: np.ndarray
    anchor_profiles: np.ndarray
    anchor_amplitudes: np.ndarray
    diagnostics: dict

@lru_cache(maxsize=8)
def multiscale_curvature(n):
    blocks=[]
    for step in (1,4,16):
        if 2*step>=n:continue
        row=np.arange(n-2*step)
        a=np.zeros((len(row),n));a[row,row]=1/step**2
        a[row,row+step]=-2/step**2;a[row,row+2*step]=1/step**2
        blocks.append(a)
    return np.vstack(blocks)

def positive_qp(P,q,tolerance=1e-8):
    """min 0.5*x'P*x+q'x, x>=0, with primal/dual diagnostics."""
    try:
        R=cholesky(P,lower=False)
        y=solve_triangular(R.T,-q,lower=True)
        v,_=nnls(R,y,maxiter=20000)
        gradient=P@v+q
        pg=float(abs(v-np.maximum(v-gradient,0)).max())
        scaled=pg/max(1.,float(abs(q).max()))
        if scaled<=20*tolerance:
            return v,dict(status='NNLS KKT',converged=True,optimizer='Cholesky/active-set NNLS',
                projected_gradient_inf=pg,scaled_projected_gradient=scaled,
                complementarity=float(abs(v@gradient)),minimum_before_roundoff_clip=float(v.min()))
    except (np.linalg.LinAlgError,RuntimeError):
        pass
    n=len(q);s=clarabel.DefaultSettings();s.verbose=False;s.max_iter=250
    s.tol_gap_abs=tolerance;s.tol_gap_rel=tolerance;s.tol_feas=tolerance*.1
    solver=clarabel.DefaultSolver(sparse.csc_matrix(np.triu(P)),np.asarray(q),
        -sparse.eye(n,format='csc'),np.zeros(n),[clarabel.NonnegativeConeT(n)],s)
    solution=solver.solve();v=np.asarray(solution.x);d=solver.get_info()
    if not np.isfinite(v).all():raise FloatingPointError('Nonfinite QP result')
    diag=dict(status=str(solution.status),converged=str(solution.status)=='Solved',
              iterations=solution.iterations,primal_residual=float(d.res_primal),
              dual_residual=float(d.res_dual),gap_abs=float(d.gap_abs),gap_rel=float(d.gap_rel),
              minimum_before_roundoff_clip=float(v.min()))
    if v.min() < -1e-7:raise FloatingPointError('QP returned material negative mass')
    return np.maximum(v,0),diag

def _path(K,y,sigma,L,config):
    nn,rmin=nnls(K,y,maxiter=20000)
    target=rmin*rmin+config.discrepancy_slack*np.sum(sigma*sigma)
    Q=L.T@L;P=K.T@K;q=-K.T@y;path=[]
    # Fixed geometric bounds and iterations. No distribution truth is queried.
    lo,hi=-8.,8.;best=None
    for loglam in [lo,hi]:
        lam=10.**loglam;x,d=positive_qp(P+lam*Q+1e-12*np.eye(len(q)),q,config.tolerance)
        rss=float(np.sum((K@x-y)**2));path.append(dict(lambda_value=lam,rss=rss,**d))
        if rss<=target:best=(lam,x,d)
    if best is not None and best[0]==10.**hi:return (*best,path,Q,target,rmin)
    for _ in range(config.path_steps):
        mid=(lo+hi)/2;lam=10.**mid
        x,d=positive_qp(P+lam*Q+1e-12*np.eye(len(q)),q,config.tolerance)
        rss=float(np.sum((K@x-y)**2));path.append(dict(lambda_value=lam,rss=rss,**d))
        if rss<=target:lo=mid;best=(lam,x,d)
        else:hi=mid
    if best is None:
        # Record an explicit NNLS fallback rather than call an infeasible path adequate.
        grad=K.T@(K@nn-y);d=dict(converged=True,status='NNLS fallback',
            projected_gradient_inf=float(abs(nn-np.maximum(nn-grad,0)).max()))
        best=(0.,nn,d)
    return (*best,path,Q,target,rmin)

def _one(K,y,sigma,config):
    L=multiscale_curvature(K.shape[1]);lam,x,d,path,Q,target,rmin=_path(K,y,sigma,L,config)
    pilot=x.copy();allpaths=[path]
    if config.adaptation>0:
        # Curvature envelopes at three scales, with a nonzero floor, prevent
        # division by tiny tail curvature. This is an UPEN-inspired heuristic.
        curvature=abs(L@pilot);envelope=maximum_filter1d(curvature,size=5)
        floor=max(.1*float(envelope.max()),1e-8)
        weights=(floor/(floor+envelope))**config.adaptation
        weights=np.clip(weights,.05,1.)
        lam,x,d,path,Q,target,rmin=_path(K,y,sigma,L*np.sqrt(weights[:,None]),config)
        allpaths.append(path)
    return x,dict(lambda_value=lam,target_rss=target,nnls_minimum=rmin,
                  selected=d,paths=allpaths),Q

def _anchors(K,Y,sigma,scales,independent,config):
    normalized=Y/scales;noise=sigma/scales
    pool=np.flatnonzero(scales>=.25*max(scales))
    if len(pool)<2:return np.empty((K.shape[1],0)),[]
    # SPA-like selection of observed attenuation directions. Anchors are
    # candidates from data, not supplied pure chemical spectra.
    features=normalized[:,pool];u,s,vh=np.linalg.svd(features,full_matrices=False)
    rank=min(config.max_anchors,len(s));features=s[:rank,None]*vh[:rank]
    residual=features.copy();chosen=[];library=[];metadata=[]
    for _ in range(min(config.max_anchors,len(pool))):
        norms=np.sum(residual**2,axis=0);norms[chosen]=-1
        local=int(np.argmax(norms))
        if norms[local]<=1e-12:break
        chosen.append(local);j=int(pool[local]);direction=residual[:,local].copy()
        direction/=np.linalg.norm(direction);residual-=direction[:,None]*(direction@residual)
        distances=np.sum((normalized-normalized[:,j,None])**2/(noise**2+noise[:,j,None]**2),axis=0)
        members=np.flatnonzero((distances<=chi2.ppf(.9,max(1,len(K)-1)))&(scales>=.25*scales[j]))
        if len(members)==0:members=np.array([j])
        precision=1/noise[:,members]**2
        pooled=np.sum(precision*normalized[:,members],axis=1)/precision.sum(axis=1)
        sd=np.sqrt(1/precision.sum(axis=1))
        profile,diag,_=_one(K,pooled,sd,config)
        total=profile.sum()
        if total<=1e-12:continue
        library.append(profile/total)
        metadata.append(dict(anchor_column=j,pooled_columns=members.tolist(),fit=diag,
            uncertainty_note='Pooling threshold is heuristic because amplitudes and membership use observed data'))
    return np.column_stack(library) if library else np.empty((K.shape[1],0)),metadata

def constrained_sharing(P,H,y,radius,tolerance):
    """Minimize conditional regularization subject to the same residual budget."""
    n,m=H.shape;s=clarabel.DefaultSettings();s.verbose=False;s.max_iter=300
    s.tol_gap_abs=tolerance;s.tol_gap_rel=tolerance;s.tol_feas=tolerance*.1
    A=sparse.vstack([-sparse.eye(m),sparse.csc_matrix((1,m)),-sparse.csc_matrix(H)],format='csc')
    rhs=np.r_[np.zeros(m),radius,-y]
    solver=clarabel.DefaultSolver(sparse.csc_matrix(np.triu(P)),np.zeros(m),A,rhs,
        [clarabel.NonnegativeConeT(m),clarabel.SecondOrderConeT(n+1)],s)
    solution=solver.solve();v=np.asarray(solution.x);d=solver.get_info()
    if not np.isfinite(v).all() or v.min()<-1e-7:raise FloatingPointError('Invalid constrained sharing output')
    v=np.maximum(v,0);violation=max(0.,float(np.linalg.norm(H@v-y)-radius))
    return v,dict(status=str(solution.status),converged=bool(str(solution.status)=='Solved' and violation<=1e-6),
        optimizer='Clarabel SOC-constrained quadratic',iterations=solution.iterations,
        primal_residual=float(d.res_primal),dual_residual=float(d.res_dual),gap_abs=float(d.gap_abs),
        gap_rel=float(d.gap_rel),ball_violation=violation,noise_radius=float(radius))

def _solve(problem,config,joint):
    edges=np.geomspace(*problem.rate_bounds,config.bins+1);D=np.sqrt(edges[:-1]*edges[1:])
    K=np.exp(-problem.b[:,None]*D);Y=problem.y;scales=np.maximum(Y[0],5*np.max(problem.sigma,axis=0))
    columns=[];diags=[];Qs=[]
    for j in range(Y.shape[1]):
        x,d,Q=_one(K,Y[:,j]/scales[j],problem.sigma[:,j]/scales[j],config)
        columns.append(x);diags.append(d);Qs.append(Q)
    independent=np.column_stack(columns);x=independent.copy();shared=np.zeros_like(x);private=x.copy()
    U=np.empty((config.bins,0));amplitudes=np.empty((0,Y.shape[1]));metadata=[];joint_diags=[]
    if joint and Y.shape[1]>1 and config.sharing>0:
        U,metadata=_anchors(K,Y,problem.sigma,scales,independent,config)
        if U.shape[1]:
            C=np.column_stack([U,np.eye(config.bins)]);H=K@C;k=U.shape[1]
            amplitudes=np.zeros((k,Y.shape[1]))
            for j in range(Y.shape[1]):
                noise=problem.sigma[:,j]/scales[j]
                gamma=config.sharing*np.sum(noise**2)/max(np.sum(independent[:,j]**2),1e-12)
                P=diags[j]['lambda_value']*(C.T@Qs[j]@C)
                P[k:,k:]+=gamma*np.eye(config.bins);P+=1e-12*np.eye(len(P))
                if config.constrain_sharing:
                    theta,d=constrained_sharing(P,H,Y[:,j]/scales[j],np.sqrt(diags[j]['target_rss']),config.tolerance)
                else:
                    theta,d=positive_qp(P+H.T@H,-H.T@(Y[:,j]/scales[j]),config.tolerance)
                amplitudes[:,j]=theta[:k];shared[:,j]=U@theta[:k];private[:,j]=theta[k:]
                x[:,j]=shared[:,j]+private[:,j];d['private_penalty']=gamma;joint_diags.append(d)
    x*=scales;shared*=scales;private*=scales;independent*=scales;amplitudes*=scales
    solved=[d['selected']['converged'] for d in diags]+[d['fit']['selected']['converged'] for d in metadata]+[d['converged'] for d in joint_diags]
    diag=dict(method='RAI multiscale shared/private' if joint else 'RAI multiscale single',config=asdict(config),
        converged=bool(all(solved)),columns=diags,anchors=metadata,joint_columns=joint_diags,
        noise_adequacy_calibrated=False,component_types_identified=False,chemical_species_identified=False,
        objective_convex_conditional_on_weights_and_anchors=True,global_adaptive_convergence_proved=False,
        scientific_status='experimental hypothesis; no novelty or superiority established')
    return MultiscaleResult(edges,x,K@x,independent,shared,private,U,amplitudes,diag)

def solve_multiscale_single(problem:DecayProblem,config=MultiscaleConfig()):
    if problem.y.shape[1]!=1:raise ValueError('Single solver requires one attenuation curve')
    return _solve(problem,config,False)

def solve_multiscale_dosy(problem:DecayProblem,config=MultiscaleConfig()):
    return _solve(problem,config,True)
