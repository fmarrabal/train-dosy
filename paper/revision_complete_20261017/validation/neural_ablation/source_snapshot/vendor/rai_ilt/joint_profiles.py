"""Research candidate: jointly learned positive spline profiles and point masses.

Ordinary exp(-bD) attenuation. Spectra are eliminated by nonnegative least
squares at every profile update (variable projection). Smoothness acts only
on continuous densities. Neither model type, chemical identity nor a global
optimum is certified. The selection score is a local effective-df heuristic,
not a calibrated likelihood test for these nonregular mixtures.
"""
from dataclasses import dataclass,asdict
import numpy as np
from scipy.optimize import minimize,nnls
from scipy.special import softmax
from scipy import sparse
import clarabel
from .model import DecayProblem
from .continuous_unmixing import ContinuousConfig,spline_operator

@dataclass(frozen=True)
class JointProfileConfig:
    bins:int=256
    basis_size:int=24
    max_factors:int=6
    starts:int=2
    smoothness:float=.01
    max_iterations:int=450
    ftol:float=1e-10
    gtol:float=1e-6
    seed:int=95111
    active_fraction:float=1e-5
    refinement_sweeps:int=40
    def __post_init__(self):
        for key,low in [('bins',128),('basis_size',8),('max_factors',1),('starts',1),('max_iterations',1),('seed',0),('refinement_sweeps',0)]:
            v=getattr(self,key)
            if isinstance(v,bool) or not isinstance(v,(int,np.integer)) or v<low:raise ValueError(key)
        if self.basis_size>self.bins or self.max_factors>10:raise ValueError('Unsupported basis or rank limit')
        for key in ('smoothness','ftol','gtol','active_fraction'):
            if not np.isfinite(getattr(self,key)) or getattr(self,key)<=0:raise ValueError(key)

class JointOperator:
    def __init__(self,problem,config):
        basis=spline_operator(problem.b,problem.rate_bounds,ContinuousConfig(
            bins=config.bins,basis_size=config.basis_size,quadrature_order=8))
        integrals=basis['mass_basis'].sum(axis=0)
        self.G=basis['operator']/integrals
        self.L=basis['roughness']/integrals
        self.M=basis['mass_basis']/integrals
        self.F=basis['density_basis']/integrals
        self.edges=basis['edges'];self.rates=basis['rates']
        self.origin=float(np.mean(np.log(problem.rate_bounds)))
        self.bounds=np.log(problem.rate_bounds)-self.origin
        self.b=problem.b;self.m=config.basis_size
        self.basis_centers=np.log(self.rates/np.exp(self.origin))@self.M

    def unpack(self,theta,atomic,continuous):
        rates=np.exp(theta[:atomic]+self.origin)
        if continuous:
            logits=theta[atomic:].reshape(self.m,continuous)
            C=softmax(logits,axis=0)
        else:C=np.empty((self.m,0))
        return rates,C

    def evaluate(self,theta,atomic,continuous):
        rates,C=self.unpack(theta,atomic,continuous)
        A=np.exp(-self.b[:,None]*rates)
        return np.column_stack([A,self.G@C]),rates,C

    def initial(self,centers,atomic,continuous,width=.35):
        if continuous:
            v=-.5*((self.basis_centers[:,None]-centers[None,atomic:])/width)**2
            # A tail basis is an unstable gauge reference for a narrow profile.
            # Keep all logits, initialize relative to their maximum, and allow
            # the harmless common-shift null direction during optimization.
            v=np.clip(v-v.max(axis=0),-24,0).ravel()
        else:v=np.empty(0)
        return np.r_[centers[:atomic],v]

def amplitudes(H,Y,sigma):
    # Acquisition-independent column scales cancel in the NNLS minimizer.
    if np.all(sigma==sigma[:1]):
        return np.column_stack([nnls(H,Y[:,j],maxiter=max(500,50*H.shape[1]))[0] for j in range(Y.shape[1])])
    return np.column_stack([nnls(H/sigma[:,j,None],Y[:,j]/sigma[:,j],maxiter=max(500,50*H.shape[1]))[0] for j in range(Y.shape[1])])

def objective(theta,op,Y,sigma,atomic,continuous,smoothness,details=False):
    H,rates,C=op.evaluate(theta,atomic,continuous)
    S=amplitudes(H,Y,sigma);R=(H@S-Y)/sigma
    rough=op.L@C;value=.5*(np.sum(R*R)+smoothness*np.sum(rough*rough))
    gradH=(R/sigma)@S.T
    ga=np.sum(gradH[:,:atomic]*(-op.b[:,None]*rates*H[:,:atomic]),axis=0)
    if continuous:
        gc=op.G.T@gradH[:,atomic:]+smoothness*(op.L.T@rough)
        gl=C*(gc-np.sum(C*gc,axis=0));gradient=np.r_[ga,gl.ravel()]
    else:gradient=ga
    return (float(value),gradient,H,S,C,rates) if details else (float(value),gradient)

def effective_df(op,H,S,C,rates,sigma,config):
    """Conditional tangent-space df after projecting active spectral amplitudes.

    This approximation ignores active-set selection and nonlinear curvature.
    It is disclosed as a heuristic, never calibrated uncertainty or evidence.
    """
    atomic=len(rates);directions=[];labels=[];blocks=[];continuous=len(C.T)
    for k in range(atomic):
        directions.append(-op.b*rates[k]*H[:,k]);labels.append(k);blocks.append(np.zeros((1,1)))
    for k in range(continuous):
        active=np.flatnonzero(C[:,k]>config.active_fraction)
        if len(active)<2:continue
        contrast=np.eye(op.m)[:,active[:-1]]-np.eye(op.m)[:,active[-1],None]
        directions.extend((op.G@contrast).T);labels.extend([atomic+k]*(len(active)-1))
        R=op.L@contrast;blocks.append(config.smoothness*(R.T@R))
    q=len(labels);G=np.zeros((q,q));P=np.zeros_like(G);offset=0
    for block in blocks:
        size=len(block);P[offset:offset+size,offset:offset+size]=block;offset+=size
    spectral_df=0
    if q:
        Jbase=np.column_stack(directions);labels=np.array(labels)
    for j in range(S.shape[1]):
        active=S[:,j]>config.active_fraction*max(float(S[:,j].max()),1e-12)
        nuisance=H[:,active]/sigma[:,j,None]
        spectral_df+=np.linalg.matrix_rank(nuisance,tol=1e-9*np.linalg.norm(nuisance)) if nuisance.size else 0
        if q:
            J=Jbase*S[labels,j]/sigma[:,j,None]
            if nuisance.size:J-=nuisance@np.linalg.lstsq(nuisance,J,rcond=1e-10)[0]
            G+=J.T@J
    if q:
        w,V=np.linalg.eigh((G+P+G.T+P.T)/2)
        keep=w>max(1e-12,1e-10*float(w.max()))
        numerator=np.sum(V[:,keep]*(G@V[:,keep]),axis=0)
        profile_df=float(np.clip(np.sum(numerator/w[keep]),0,q))
    else:profile_df=0.
    return float(spectral_df+profile_df),dict(spectral_df=int(spectral_df),profile_df=profile_df,
        tangent_parameters=q,local_linearization=True,selection_uncertainty_included=False)

def stationarity(op,H,S,C,rates,Y,sigma,smoothness):
    R=(H@S-Y)/sigma**2;a=len(rates)
    gc=op.G.T@(R@S[a:].T)+smoothness*op.L.T@(op.L@C)
    scale_c=np.maximum(1,np.max(np.abs(op.G.T@((Y/sigma**2)@S[a:].T)),axis=0)) if C.size else np.empty(0)
    values=[]
    for k in range(C.shape[1]):
        active=C[:,k]>1e-7;nu=float(np.mean(gc[active,k]))
        residual=max(float(np.max(np.abs(gc[active,k]-nu))),
                     float(np.max(np.maximum(nu-gc[~active,k],0))) if (~active).any() else 0.)
        values.append(residual/scale_c[k])
    if a:
        J=-op.b[:,None]*rates*H[:,:a];ga=np.sum(J*(R@S[:a].T),axis=0)
        z=np.log(rates)-op.origin
        pg=np.where(z<=op.bounds[0]+1e-8,np.maximum(-ga,0),
                    np.where(z>=op.bounds[1]-1e-8,np.maximum(ga,0),np.abs(ga)))
        scale=np.maximum(1,np.sum(np.abs(J)*((np.abs(Y)/sigma**2)@S[:a].T),axis=0))
        values.extend(pg/scale)
    gs=H.T@R;scale=np.maximum(1,np.max(np.abs(H.T@(Y/sigma**2)),axis=0))
    # True nonnegative KKT residual; do not cap a negative free gradient by S.
    active=S>1e-8*np.maximum(S.max(axis=0),1e-12)
    residual=np.where(active,np.abs(gs),np.maximum(-gs,0))
    values.extend(np.max(residual,axis=0)/scale)
    return float(max(values,default=0.))

def refine_profiles(op,H,S,C,rates,Y,sigma,config):
    """Monotone block refinement in density coordinates, outside softmax gauge."""
    C=C.copy();rates=rates.copy();a=len(rates);Q=op.L.T@op.L
    def score(H,S,C):return .5*float(np.sum(((H@S-Y)/sigma)**2)+config.smoothness*np.sum((op.L@C)**2))
    history=[score(H,S,C)];statuses=[];accepted=0
    s=clarabel.DefaultSettings();s.verbose=False;s.max_iter=150
    s.tol_gap_abs=1e-10;s.tol_gap_rel=1e-10;s.tol_feas=1e-10
    m=op.m;constraints=sparse.vstack([-sparse.eye(m),np.ones((1,m))],format='csc')
    rhs=np.r_[np.zeros(m),1.];cones=[clarabel.NonnegativeConeT(m),clarabel.ZeroConeT(1)]
    for sweep in range(config.refinement_sweeps):
        for k in range(C.shape[1]):
            r=a+k;residual=Y-H@S+H[:,r,None]*S[r]
            precision=np.sum(S[r]**2/sigma**2,axis=1)
            linear=np.sum(S[r]*residual/sigma**2,axis=1)
            P=op.G.T@(precision[:,None]*op.G)+config.smoothness*Q;q=-op.G.T@linear
            scale=max(1.,float(np.max(np.abs(P))),float(np.max(np.abs(q))))
            solver=clarabel.DefaultSolver(sparse.csc_matrix(np.triu(P/scale)),q/scale,
                constraints,rhs,cones,s);solution=solver.solve();statuses.append(str(solution.status))
            c=np.asarray(solution.x)
            if not np.isfinite(c).all() or c.min()<-1e-7 or abs(c.sum()-1)>1e-6:continue
            c=np.maximum(c,0);c/=c.sum()
            trialC=C.copy();trialC[:,k]=c;trialH=H.copy();trialH[:,r]=op.G@c
            trialS=amplitudes(trialH,Y,sigma);value=score(trialH,trialS,trialC)
            if value<=history[-1]:H,S,C=trialH,trialS,trialC;history.append(value);accepted+=1
        if a:
            def atomic_fun(z,details=False):
                atom=np.exp(-op.b[:,None]*np.exp(z+op.origin));hh=np.column_stack([atom,op.G@C])
                ss=amplitudes(hh,Y,sigma);rr=(hh@ss-Y)/sigma
                gradient=np.sum((-op.b[:,None]*np.exp(z+op.origin)*atom)*((rr/sigma)@ss[:a].T),axis=0)
                return (hh,ss,np.exp(z+op.origin)) if details else (.5*float(np.sum(rr*rr)),gradient)
            fit=minimize(atomic_fun,np.log(rates)-op.origin,jac=True,method='L-BFGS-B',bounds=[tuple(op.bounds)]*a,
                options=dict(maxiter=40,ftol=1e-12,gtol=1e-7,maxls=30))
            hh,ss,rr=atomic_fun(fit.x,True);value=score(hh,ss,C)
            if value<=history[-1]:H,S,rates=hh,ss,rr;history.append(value);accepted+=1
        residual=stationarity(op,H,S,C,rates,Y,sigma,config.smoothness)
        if residual<=1e-6:break
    residual=stationarity(op,H,S,C,rates,Y,sigma,config.smoothness)
    counts={v:statuses.count(v) for v in sorted(set(statuses))}
    return H,S,C,rates,dict(sweeps=sweep+1 if config.refinement_sweeps else 0,
        accepted_steps=accepted,objective_history=history,conditional_qp_statuses=counts,
        scaled_kkt_residual=residual,locally_stationary=residual<=1e-6,
        global_optimum_certified=False)

@dataclass
class JointProfileResult:
    rate_edges:np.ndarray
    atomic_rates:np.ndarray
    atomic_spectra:np.ndarray
    continuous_profiles:np.ndarray
    continuous_spectra:np.ndarray
    continuous_coefficients:np.ndarray
    atomic_masses:np.ndarray
    continuous_masses:np.ndarray
    density_log_rate:np.ndarray
    reconstructed:np.ndarray
    diagnostics:dict
    candidates:list
    config:dict
    rate_bounds:tuple
    @property
    def masses(self):return self.atomic_masses+self.continuous_masses
    def predict(self,b):
        b=np.asarray(b,dtype=float).reshape(-1)
        p=DecayProblem(b,np.zeros((len(b),1)),1.,self.rate_bounds)
        op=JointOperator(p,JointProfileConfig(**self.config))
        return np.exp(-b[:,None]*self.atomic_rates)@self.atomic_spectra+(op.G@self.continuous_coefficients)@self.continuous_spectra

def _recover(problem,config,joint):
    if not joint and problem.y.shape[1]!=1:raise ValueError('Single mode requires one curve')
    op=JointOperator(problem,config);Y=problem.y;sigma=problem.sigma
    lo,hi=op.bounds;rng=np.random.default_rng(config.seed)
    fine=np.linspace(lo,hi,128);kernel=np.exp(-problem.b[:,None]*np.exp(fine+op.origin))
    pooled=Y.mean(axis=1);initial_mass,_=nnls(kernel,pooled,maxiter=10000)
    cdf=np.cumsum(initial_mass)/max(initial_mass.sum(),1e-300)
    candidates=[];fits=[]
    N=Y.size
    for rank in range(1,config.max_factors+1):
        for continuous in range((rank if joint else min(rank,1))+1):
            atomic=rank-continuous
            bounds=[(lo,hi)]*atomic+[(-30.,30.)]*(op.m*continuous)
            guesses=[]
            for start in range(config.starts):
                quantiles=(np.arange(rank)+.5)/rank
                center=np.interp(quantiles,cdf,fine) if start==0 and initial_mass.sum()>0 else np.linspace(lo,hi,rank+2)[1:-1]
                center=np.clip(np.roll(center,start)+rng.normal(0,.08,rank),lo,hi)
                guesses.append(op.initial(center,atomic,continuous,width=(.3,.6,.18)[start%3]))
            optimized=[]
            for guess in guesses:
                fit=minimize(objective,guess,args=(op,Y,sigma,atomic,continuous,config.smoothness),
                    method='L-BFGS-B',jac=True,bounds=bounds,
                    options=dict(maxiter=config.max_iterations,ftol=config.ftol,gtol=config.gtol,maxls=40))
                optimized.append(fit)
            # A budget exit must not silently replace a better physical fit
            # with an inferior model. Keep the best finite fit and its status.
            best=min([fit for fit in optimized if np.isfinite(fit.fun)],key=lambda fit:fit.fun)
            value,g,H,S,C,rates=objective(best.x,op,Y,sigma,atomic,continuous,config.smoothness,True)
            H,S,C,rates,refinement=refine_profiles(op,H,S,C,rates,Y,sigma,config)
            value=refinement['objective_history'][-1]
            df,dfdiag=effective_df(op,H,S,C,rates,sigma,config)
            rss=float(np.sum(((H@S-Y)/sigma)**2))
            score=rss+2*df+2*df*(df+1)/max(1,N-df-1)
            pg=best.x-np.clip(best.x-g,np.array(bounds)[:,0],np.array(bounds)[:,1])
            row=dict(atomic=atomic,continuous=continuous,rank=rank,rss=rss,objective=value,
                score=float(score),effective_df=df,df_diagnostics=dfdiag,
                native_success=bool(best.success),message=str(best.message),iterations=int(best.nit),
                starts_successful=sum(bool(f.success) for f in optimized),
                starts=[dict(success=bool(f.success),objective=float(f.fun),iterations=int(f.nit)) for f in optimized],
                projected_gradient_inf=float(np.max(np.abs(pg))),
                scaled_projected_gradient=float(np.max(np.abs(pg))/max(1,value)),
                refinement=refinement,theta_initial_optimizer=best.x.tolist(),spectra=S.tolist(),atomic_rates=rates.tolist(),
                continuous_coefficients=C.tolist())
            candidates.append(row);fits.append((H,S,C,rates))
    eligible=[i for i,r in enumerate(candidates) if np.isfinite(r['score'])]
    search_failed=not eligible
    selected=min(eligible or range(len(candidates)),key=lambda i:candidates[i]['score'])
    row=candidates[selected];H,S,C,rates=fits[selected];a=row['atomic']
    atom_mass=np.zeros((config.bins,len(rates)))
    if len(rates):atom_mass[np.clip(np.searchsorted(op.edges,rates)-1,0,config.bins-1),np.arange(len(rates))]=1.
    continuous_profiles=op.M@C
    for i,r in enumerate(candidates):r.update(selected=i==selected,delta_score=r['score']-row['score'])
    competitive=[i for i,r in enumerate(candidates) if r['delta_score']<=2]
    diag=dict(method='RAI jointly learned flexible profiles experimental',mode='dosy' if joint else 'single',
        selected_candidate=selected,atomic_factors=a,continuous_factors=row['continuous'],
        native_success=row['native_success'],search_failed=search_failed,
        search_complete=all(r['native_success'] for r in candidates),
        scaled_projected_gradient=row['scaled_projected_gradient'],
        locally_stationary=row['refinement']['locally_stationary'],
        scaled_kkt_residual=row['refinement']['scaled_kkt_residual'],
        competitive_candidates=competitive,type_ambiguous=len(competitive)>1,
        selection='RSS + 2 local effective df + finite-sample correction; heuristic',
        spectrum_and_profiles_jointly_updated=True,fixed_anchors=False,
        single_continuous_factors_collapsed=not joint,
        noise_adequacy_calibrated=False,global_optimality_proved=False,
        chemical_species_identified=False,scientific_novelty_established=False)
    return JointProfileResult(op.edges,rates,S[:a],continuous_profiles,S[a:],C,
        atom_mass@S[:a],continuous_profiles@S[a:],op.F@C@S[a:],H@S,diag,
        candidates,asdict(config),problem.rate_bounds)

def solve_joint_single(problem:DecayProblem,config=JointProfileConfig()):
    return _recover(problem,config,False)

def solve_joint_dosy(problem:DecayProblem,config=JointProfileConfig()):
    if problem.ppm is None:raise ValueError('DOSY requires an explicit frequency coordinate')
    # Canonical evaluation order avoids finite-iteration changes caused solely
    # by floating-point summation order. It imposes no frequency-neighbor prior.
    order=np.argsort(problem.ppm,kind='stable');back=np.argsort(order)
    p=DecayProblem(problem.b,problem.y[:,order],problem.sigma[:,order],problem.rate_bounds,problem.ppm[order])
    result=_recover(p,config,True)
    for key in ('atomic_spectra','continuous_spectra','atomic_masses','continuous_masses','density_log_rate','reconstructed'):
        setattr(result,key,getattr(result,key)[:,back])
    for row in result.candidates:row['spectra']=np.asarray(row['spectra'])[:,back].tolist()
    return result
