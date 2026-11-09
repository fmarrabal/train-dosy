"""Active-set dual refinement only when the square-root certificate fails."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
from scipy.linalg import qr
from scipy.optimize import nnls,brentq
from .stable_profiles import solve_stable_dosy,_one_factor,optimality_bound
from .multiscale_profiles import MultiscaleConfig


def refine_conditional(F,H,y,radius,previous,tolerance=1e-8):
    """Solve the same convex problem via scalar dual search and QR/NNLS.

    For mu>0, NNLS minimizes ||F theta||² + mu ||H theta-y||².
    Its residual norm is monotone in mu. A bracketed root enforces the original
    radius. The same independent bound, not root status, decides acceptance.
    """
    orthogonal,R=qr(F,mode='economic',check_finite=False)
    mu=max(previous['norm_objective']*previous['norm_multiplier']/radius,1e-12)
    center=float(np.log(mu));calls=[]
    def solve(logmu):
        weight=np.exp(logmu/2)
        x,_=nnls(np.vstack([R,weight*H]),np.r_[np.zeros(R.shape[0]),weight*y],maxiter=20000)
        residual=float(np.linalg.norm(H@x-y))
        calls.append(dict(log_multiplier=float(logmu),residual_norm=residual))
        return x,residual
    lo=center-3.;hi=center+3.
    for _ in range(12):
        _,rlo=solve(lo);_,rhi=solve(hi)
        if rlo>=radius>=rhi:break
        if rlo<radius:lo-=3.
        if rhi>radius:hi+=3.
    else:raise RuntimeError('Unable to bracket the original residual budget')
    root=brentq(lambda t:solve(t)[1]-radius,lo,hi,xtol=1e-12,rtol=1e-14,maxiter=100)
    x,residual=solve(root);norm=float(np.linalg.norm(F@x))
    p=R@x/max(norm,1e-300);w=np.exp(root)*(H@x-y)/max(norm,1e-300)
    d=optimality_bound(F,H,y,radius,x,float(np.linalg.norm(w)),orthogonal@p,w)
    accepted=bool(d['relative_objective_gap']<=tolerance and d['relative_ball_violation']<=tolerance
        and d['lower_bound_overshoot']<=tolerance and x.min()>=-tolerance)
    return x,dict(**d,converged=accepted,status='Bound certified' if accepted else 'Uncertified',
        native_status='NNLS/Brent refinement',optimizer='Square-root NNLS with scalar dual root',
        dual_factor_compressed=p.tolist(),certificate_residual_vector=w.tolist(),
        attempts=[dict(optimizer='NNLS/Brent',calls=len(calls),converged=accepted)],dual_path=calls,
        certificate_scope='same conditional convex problem; floating-point bound',
        previous_square_root_diagnostic=previous)


def refine_result(problem,result):
    """Refine a result produced by solve_stable_dosy for this exact problem."""
    bad=[j for j,d in enumerate(result.diagnostics['joint_columns']) if not d['converged']]
    if not bad:return result
    config=MultiscaleConfig(**result.diagnostics['config']);diagnostics=deepcopy(result.diagnostics)
    r=replace(result,masses=result.masses.copy(),reconstructed=result.reconstructed.copy(),
        shared_masses=result.shared_masses.copy(),private_masses=result.private_masses.copy(),
        anchor_amplitudes=result.anchor_amplitudes.copy(),diagnostics=diagnostics)
    D=np.sqrt(r.rate_edges[:-1]*r.rate_edges[1:]);K=np.exp(-problem.b[:,None]*D)
    scales=np.maximum(problem.y[0],5*np.max(problem.sigma,axis=0));U=r.anchor_profiles;k=U.shape[1]
    C=np.column_stack([U,np.eye(config.bins)]);H=K@C;revised=[]
    for j in bad:
        y=problem.y[:,j]/scales[j];noise=problem.sigma[:,j]/scales[j]
        independent,diag,L=_one_factor(K,y,noise,config)
        np.testing.assert_allclose(independent*scales[j],r.independent_masses[:,j],atol=1e-12,rtol=1e-12)
        gamma=config.sharing*np.sum(noise**2)/max(np.sum(independent**2),1e-12)
        F=np.vstack([np.sqrt(diag['lambda_value'])*(L@C),
            np.sqrt(gamma)*np.column_stack([np.zeros((config.bins,k)),np.eye(config.bins)]),1e-6*np.eye(C.shape[1])])
        try:
            theta,d=refine_conditional(F,H,y,np.sqrt(diag['target_rss']),diagnostics['joint_columns'][j],config.tolerance)
            d['private_penalty']=gamma
            if not d['converged']:
                diagnostics['joint_columns'][j]['refinement_attempt']=d;continue
            r.anchor_amplitudes[:,j]=theta[:k]*scales[j]
            r.shared_masses[:,j]=U@theta[:k]*scales[j];r.private_masses[:,j]=theta[k:]*scales[j]
            r.masses[:,j]=r.shared_masses[:,j]+r.private_masses[:,j];r.reconstructed[:,j]=K@r.masses[:,j]
            diagnostics['joint_columns'][j]=d;revised.append(j)
        except (RuntimeError,np.linalg.LinAlgError) as e:
            diagnostics['joint_columns'][j]['refinement_error']=str(e)
    diagnostics['numerical_refinement']=dict(attempted_columns=bad,revised_columns=revised,
        model_changed=False,tolerance_relaxed=False)
    diagnostics['converged']=bool(all(d['selected']['converged'] for d in diagnostics['columns'])
        and all(d['fit']['selected']['converged'] for d in diagnostics['anchors'])
        and all(d['converged'] for d in diagnostics['joint_columns']))
    return r


def solve_refined_dosy(problem,config=MultiscaleConfig()):
    return refine_result(problem,solve_stable_dosy(problem,config))
