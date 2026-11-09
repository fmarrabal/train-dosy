"""Square-root numerical implementation of the frozen multiscale DOSY model.

The prior, anchors, residual budgets and hyperparameters are unchanged. The
conditional problem is solved in norm form without forming its Gram matrix.
A convex supporting-plane lower bound independently checks optimality.
"""
from dataclasses import asdict
import numpy as np
from scipy import sparse
from scipy.linalg import qr
from scipy.ndimage import maximum_filter1d
import clarabel
from . import multiscale_profiles as legacy
from .multiscale_profiles import MultiscaleConfig,MultiscaleResult,solve_multiscale_single


def _one_factor(K,y,sigma,config):
    """Exactly the legacy paths, retaining the regularizer before squaring it."""
    L=legacy.multiscale_curvature(K.shape[1])
    lam,x,d,path,Q,target,rmin=legacy._path(K,y,sigma,L,config)
    allpaths=[path]
    if config.adaptation>0:
        envelope=maximum_filter1d(abs(L@x),size=5)
        floor=max(.1*float(envelope.max()),1e-8)
        weights=np.clip((floor/(floor+envelope))**config.adaptation,.05,1.)
        L=L*np.sqrt(weights[:,None])
        lam,x,d,path,Q,target,rmin=legacy._path(K,y,sigma,L,config)
        allpaths.append(path)
    return x,dict(lambda_value=lam,target_rss=target,nnls_minimum=rmin,
                  selected=d,paths=allpaths),L


def optimality_bound(F,H,y,radius,x,multiplier,dual_factor=None,dual_residual=None):
    """Numerical lower bound for min ||F x|| with x>=0, ||H x-y||<=radius.

    For any feasible u, ||u||_1 <= B from a strictly positive row of H.
    Convexity of ||F u|| + multiplier*(||H u-y||-radius) gives
    lower = f + multiplier*(r-radius) - g'x + B*min(0,min(g)).
    Clipping at zero uses the nonnegative objective. This is a supporting-
    plane bound, not a calibrated statistical confidence interval. Computed
    in floating point; no interval-arithmetic or global adaptive certificate.
    """
    fx=F@x;res=H@x-y;f=float(np.linalg.norm(fx));r=float(np.linalg.norm(res))
    positive=np.min(H,axis=1)>0
    if not positive.any():raise ValueError('A strictly positive kernel row is required for the certificate')
    B=float(np.min((y[positive]+radius)/np.min(H[positive],axis=1)))
    if B<0:raise ValueError('Residual ball is incompatible with a nonnegative signal')
    multiplier=max(0.,float(multiplier))
    g=F.T@(fx/max(f,1e-300))+multiplier*(H.T@(res/max(r,1e-300)))
    if dual_factor is None:
        lower=max(0.,float(f+multiplier*(r-radius)-g@x+B*min(0.,float(g.min()))))
    else:
        # Any ||p||<=1 and any w give the global lower bound
        # B*min(0,min(F.T@p+H.T@w)) - w@y - radius*||w||.
        # Using the solver's dual directions avoids the square-root loss of
        # precision incurred by reconstructing them from a nearly optimal x.
        p=np.asarray(dual_factor);p=p/max(1.,float(np.linalg.norm(p)))*(1-4*np.finfo(float).eps)
        w=np.asarray(dual_residual);g=F.T@p+H.T@w
        lower=max(0.,float(B*min(0.,float(g.min()))-w@y-radius*np.linalg.norm(w)))
    objective=.5*f*f;lower_q=.5*lower*lower
    return dict(objective=objective,objective_lower_bound=lower_q,
        objective_gap=max(0.,objective-lower_q),relative_objective_gap=max(0.,objective-lower_q)/max(1.,objective),
        lower_bound_overshoot=max(0.,lower_q-objective)/max(1.,objective),
        ball_violation=max(0.,r-radius),relative_ball_violation=max(0.,r-radius)/max(1.,radius),
        minimum_mass=float(x.min()),l1_upper_bound=B,norm_multiplier=multiplier,
        norm_objective=f,norm_lower_bound=lower,
        projected_stationarity=float(np.max(abs(np.minimum(g,x)))),
        complementarity=float(abs(g@x)),noise_radius=float(radius))


def constrained_square_root(F,H,y,radius,feasible,tolerance=1e-8):
    """Same quadratic minimizer, norm epigraph and independent objective bound."""
    F=np.asarray(F,dtype=float);H=np.asarray(H,dtype=float);y=np.asarray(y,dtype=float)
    m=H.shape[1]
    if (F.ndim!=2 or F.shape[1]!=m or y.shape!=(H.shape[0],)
        or not all(np.isfinite(a).all() for a in (F,H,y,feasible))
        or not np.isfinite(radius) or radius<=0 or not np.isfinite(tolerance) or tolerance<=0):
        raise ValueError('Invalid square-root constrained problem')
    if np.linalg.norm(y)<=radius:
        x=np.zeros(m);d=optimality_bound(F,H,y,radius,x,0.)
        return x,dict(**d,status='Exact zero',native_status='not called',converged=True,attempts=[])
    # Orthogonal compression preserves the norm and the small ridge. It does
    # not square the condition number as F.T@F would.
    orthogonal,R=qr(F,mode='economic',check_finite=False)
    objective_scale=max(float(np.linalg.norm(F@feasible)),1e-12)
    residual_scale=max(float(radius),1e-12)
    rows=R.shape[0];n=len(y)
    A=sparse.vstack([sparse.hstack([-sparse.eye(m),np.zeros((m,1))]),
        sparse.csc_matrix(np.r_[np.zeros(m),-1][None,:]),
        sparse.hstack([-sparse.csc_matrix(R/objective_scale),np.zeros((rows,1))]),
        sparse.csc_matrix((1,m+1)),
        sparse.hstack([-sparse.csc_matrix(H/residual_scale),np.zeros((n,1))])],format='csc')
    b=np.r_[np.zeros(m+rows+1),radius/residual_scale,-y/residual_scale]
    attempts=[];best=None
    for numerical_tolerance,regularization in [(min(1e-10,tolerance*.01),1e-8),(min(1e-12,tolerance*.0001),1e-10)]:
        s=clarabel.DefaultSettings();s.verbose=False;s.max_iter=300
        s.tol_gap_abs=numerical_tolerance;s.tol_gap_rel=numerical_tolerance;s.tol_feas=numerical_tolerance
        s.static_regularization_constant=regularization
        s.iterative_refinement_max_iter=20
        solver=clarabel.DefaultSolver(sparse.csc_matrix((m+1,m+1)),np.r_[np.zeros(m),1.],A,b,
            [clarabel.NonnegativeConeT(m),clarabel.SecondOrderConeT(rows+1),clarabel.SecondOrderConeT(n+1)],s)
        sol=solver.solve();info=solver.get_info();raw=np.asarray(sol.x[:m])
        if not np.isfinite(raw).all():
            attempts.append(dict(native_status=str(sol.status),valid=False));continue
        x=np.maximum(raw,0.)
        mu=float(sol.z[m+rows+1])*objective_scale/residual_scale
        p_compressed=-np.asarray(sol.z[m+1:m+rows+1])
        w=-np.asarray(sol.z[m+rows+2:])*objective_scale/residual_scale
        d=optimality_bound(F,H,y,radius,x,mu,orthogonal@p_compressed,w)
        certified=bool(min(raw)>=-tolerance and d['relative_ball_violation']<=tolerance
            and d['relative_objective_gap']<=tolerance and d['lower_bound_overshoot']<=tolerance)
        attempt=dict(**d,native_status=str(sol.status),converged=certified,
            iterations=int(sol.iterations),primal_residual=float(info.res_primal),
            dual_residual=float(info.res_dual),native_gap_abs=float(info.gap_abs),
            numerical_tolerance=numerical_tolerance,static_regularization=regularization,
            minimum_before_roundoff_clip=float(raw.min()),
            dual_factor_compressed=p_compressed.tolist(),certificate_residual_vector=w.tolist())
        attempts.append(attempt)
        if best is None or d['relative_objective_gap']+d['relative_ball_violation']<best[1]['relative_objective_gap']+best[1]['relative_ball_violation']:
            best=(x,attempt)
        if certified:best=(x,attempt);break
    if best is None:raise FloatingPointError('No finite square-root solution')
    x,d=best
    traces=[{key:value for key,value in a.items() if key not in ('dual_factor_compressed','certificate_residual_vector')} for a in attempts]
    return x,dict(**d,status='Bound certified' if d['converged'] else 'Uncertified',
        optimizer='Clarabel square-root SOCP with independent supporting-plane bound',
        objective_scale=objective_scale,residual_scale=residual_scale,attempts=traces,
        certificate_scope='conditional convex problem; floating-point bound')


def solve_stable_dosy(problem,config=MultiscaleConfig()):
    if not config.constrain_sharing:raise ValueError('Stable DOSY requires constrain_sharing=True')
    edges=np.geomspace(*problem.rate_bounds,config.bins+1);D=np.sqrt(edges[:-1]*edges[1:])
    K=np.exp(-problem.b[:,None]*D);Y=problem.y
    scales=np.maximum(Y[0],5*np.max(problem.sigma,axis=0))
    columns=[];diags=[];Ls=[]
    for j in range(Y.shape[1]):
        x,d,L=_one_factor(K,Y[:,j]/scales[j],problem.sigma[:,j]/scales[j],config)
        columns.append(x);diags.append(d);Ls.append(L)
    independent=np.column_stack(columns);x=independent.copy();shared=np.zeros_like(x);private=x.copy()
    U=np.empty((config.bins,0));amplitudes=np.empty((0,Y.shape[1]));metadata=[];joint_diags=[]
    if Y.shape[1]>1 and config.sharing>0:
        U,metadata=legacy._anchors(K,Y,problem.sigma,scales,independent,config)
        if U.shape[1]:
            C=np.column_stack([U,np.eye(config.bins)]);H=K@C;k=U.shape[1]
            amplitudes=np.zeros((k,Y.shape[1]))
            private_selector=np.column_stack([np.zeros((config.bins,k)),np.eye(config.bins)])
            for j in range(Y.shape[1]):
                noise=problem.sigma[:,j]/scales[j]
                gamma=config.sharing*np.sum(noise**2)/max(np.sum(independent[:,j]**2),1e-12)
                F=np.vstack([np.sqrt(diags[j]['lambda_value'])*(Ls[j]@C),
                             np.sqrt(gamma)*private_selector,1e-6*np.eye(C.shape[1])])
                theta,d=constrained_square_root(F,H,Y[:,j]/scales[j],np.sqrt(diags[j]['target_rss']),
                    np.r_[np.zeros(k),independent[:,j]],config.tolerance)
                amplitudes[:,j]=theta[:k];shared[:,j]=U@theta[:k];private[:,j]=theta[k:]
                x[:,j]=shared[:,j]+private[:,j];d['private_penalty']=gamma;joint_diags.append(d)
    x*=scales;shared*=scales;private*=scales;independent*=scales;amplitudes*=scales
    solved=[d['selected']['converged'] for d in diags]+[d['fit']['selected']['converged'] for d in metadata]+[d['converged'] for d in joint_diags]
    diag=dict(method='RAI multiscale shared/private square-root',config=asdict(config),
        converged=bool(all(solved)),columns=diags,anchors=metadata,joint_columns=joint_diags,
        noise_adequacy_calibrated=False,component_types_identified=False,chemical_species_identified=False,
        objective_convex_conditional_on_weights_and_anchors=True,global_adaptive_convergence_proved=False,
        scientific_status='numerical reformulation of existing prototype; no novelty or superiority established')
    return MultiscaleResult(edges,x,K@x,independent,shared,private,U,amplitudes,diag)
