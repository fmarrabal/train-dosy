"""Check algebra in manuscript-v4; does not rerun or modify any DOSY fit.

Run from the repository root:
    python paper/scripts/check_formulations.py
These finite-dimensional checks supplement the derivations. They do not prove
global solver convergence, statistical calibration, or chemical recovery.
"""
from pathlib import Path
import json
import numpy as np


def main():
    rng = np.random.default_rng(25092026)
    checks = []

    def equal(name, left, right, tolerance=1e-10):
        error = float(np.linalg.norm(np.asarray(left) - np.asarray(right)))
        scale = max(1.0, float(np.linalg.norm(left)), float(np.linalg.norm(right)))
        relative = error / scale
        assert relative <= tolerance, (name, relative)
        checks.append(dict(name=name, relative_error=relative, tolerance=tolerance))

    def directional(name, objective, x, gradient):
        direction = rng.normal(size=x.shape)
        direction /= np.linalg.norm(direction)
        step = 1e-6
        numeric = (objective(x + step * direction) - objective(x - step * direction)) / (2 * step)
        equal(name, numeric, float(np.sum(gradient * direction)), 2e-6)

    n, q, p, rank = 11, 7, 5, 3
    b = np.linspace(0, 2, n)
    rates = np.geomspace(.2, 3, q)
    k = np.exp(-b[:, None] * rates)
    y = rng.normal(size=(n, p))
    s = rng.uniform(.2, 1, (q, rank)); s /= s.sum(axis=0)
    a = rng.uniform(.2, 1, (rank, p))
    rough = np.diff(np.eye(q), n=2, axis=0)
    lam_s, lam_a = .03, .07
    qa, ra = np.linalg.qr(a.T, mode='reduced')
    compressed = np.linalg.norm(k @ s @ ra.T - y @ qa)**2
    remainder = np.linalg.norm(y @ (np.eye(p) - qa @ qa.T))**2
    equal('TRAIn-MF QR residual decomposition', np.linalg.norm(k @ s @ a - y)**2, compressed + remainder)
    equal('TRAIn-MF Kronecker vectorization',
          (k @ s @ ra.T).ravel(order='F'), np.kron(ra, k) @ s.ravel(order='F'))
    objective = lambda ss: (np.linalg.norm(k @ ss @ a - y)**2 + lam_a * np.linalg.norm(a)**2)/(2*p) + lam_s*np.linalg.norm(rough @ ss)**2/2
    gradient = k.T @ (k @ s @ a - y) @ a.T/p + lam_s*rough.T @ rough @ s
    directional('TRAIn-MF S gradient with A fixed', objective, s, gradient)

    h = rng.uniform(.1, 1, q); eta = np.sqrt(h); yy = y[:, 0]
    train_f = lambda e: np.linalg.norm(k @ (e*e) - yy)**2
    train_g = 4*eta*(k.T @ (k @ h - yy))
    directional('TRAIn squared-parametrization gradient', train_f, eta, train_g)
    d_eta = np.diag(eta)
    exact_hessian = 8*d_eta @ k.T @ k @ d_eta + 4*np.diag(k.T @ (k @ h - yy))
    direction = rng.normal(size=q)
    gg = lambda e: 4*e*(k.T @ (k @ (e*e) - yy))
    equal('TRAIn exact versus Gauss-Newton Hessian distinction',
          (gg(eta+1e-6*direction)-gg(eta-1e-6*direction))/(2e-6), exact_hessian @ direction, 2e-6)

    weights = rng.uniform(size=(p, p)); weights = (weights+weights.T)/2; np.fill_diagonal(weights, 0)
    graph = np.diag(weights.sum(axis=1)) - weights
    inv_width = rng.uniform(.5, 2, q)
    c = rng.uniform(.1, 1, (q, p))
    graph_trace = np.sum(inv_width[:, None] * c * (c @ graph))
    pair_sum = sum(weights[i,j]*np.sum(inv_width*(c[:,i]-c[:,j])**2) for i in range(p) for j in range(i+1,p))
    equal('RAI density graph trace equals pairwise energy', graph_trace, pair_sum)
    sigma = rng.uniform(.5, 1, (n, p))
    density_f = lambda cc: np.sum(((k@cc-y)/sigma)**2)/(2*n) + .02*np.linalg.norm(rough@cc)**2/2 + 1e-5*np.sum(inv_width[:,None]*cc**2)/2 + .05*np.sum(inv_width[:,None]*cc*(cc@graph))/2
    density_g = k.T@((k@c-y)/sigma**2)/n + .02*rough.T@rough@c + 1e-5*inv_width[:,None]*c + .05*inv_width[:,None]*(c@graph)
    directional('RAI fixed-grid density gradient', density_f, c, density_g)

    m = 4
    transform, _ = np.linalg.qr(rng.normal(size=(m, m)))
    hh = []; cc = []; constants = []
    for _ in range(p):
        design = rng.normal(size=(n, m)); obs = rng.normal(size=n)
        hh.append(design.T@design + .1*np.eye(m)); cc.append(design.T@obs); constants.append(np.dot(obs,obs)/2)
    cc = np.array(cc).T
    u = rng.uniform(.1,1,(m,p)); normal = -rng.uniform(size=(m,p)); dual = rng.normal(size=(m,p)); dual[0]=0
    lam = .15
    dual[1:] *= np.minimum(1,lam*np.sqrt(p)/np.linalg.norm(dual[1:],axis=1,keepdims=True))
    target = cc-normal-transform.T@dual
    star = np.column_stack([np.linalg.solve(hh[j],target[:,j]) for j in range(p)])
    penalty = lam*np.sqrt(p)*np.linalg.norm((transform@u)[1:],axis=1).sum()
    primal = sum(.5*u[:,j]@hh[j]@u[:,j]-cc[:,j]@u[:,j]+constants[j] for j in range(p))+penalty
    lower_bound = sum(constants[j]-.5*target[:,j]@star[:,j] for j in range(p))
    gap = sum(.5*(u[:,j]-star[:,j])@hh[j]@(u[:,j]-star[:,j]) for j in range(p))+penalty-np.sum(dual*(transform@u))-np.sum(normal*u)
    equal('Adaptive RAI feasible primal-dual gap identity', gap, primal-lower_bound)
    assert gap >= 0

    mean, variance = 1.7, .02
    for fraction in [.08,.2,.5,.8,.92]:
        left = mean-np.sqrt(variance*(1-fraction)/fraction)
        right = mean+np.sqrt(variance*fraction/(1-fraction))
        equal(f'DOME exchanged mean pi={fraction}', fraction*left+(1-fraction)*right, mean)
        equal(f'DOME exchanged variance pi={fraction}', fraction*(left-mean)**2+(1-fraction)*(right-mean)**2, variance)

    decoder = rng.uniform(.1,1,(q,q)); decoder /= decoder.sum(axis=0)
    z = rng.uniform(.1,1,(q,p)); scale = rng.uniform(.7,1,p); noise=.5
    circe_f = lambda zz: np.linalg.norm((k@decoder@zz-y)/noise)**2/2 + .2*np.sum((rough@decoder@zz)**2/scale[None,:]**2)/2 + .01*np.sum(inv_width[:,None]**2*(decoder@zz)*((decoder@zz)@graph))/2
    physical = decoder@z
    circe_g = decoder.T@(k.T@(k@physical-y)/noise**2 + .2*rough.T@rough@physical/scale[None,:]**2 + .01*inv_width[:,None]**2*(physical@graph))
    directional('CIRCE v2 decoder and scaled-penalty gradient', circe_f, z, circe_g)
    equal('CIRCE decoder mass preservation', (decoder@z).sum(axis=0), z.sum(axis=0))

    root=Path(__file__).resolve().parents[2]
    result=dict(scope='Finite-dimensional algebra checks only; no solver benchmark, no model refitting, no identifiability claim.', seed=25092026, numpy=np.__version__, checks=checks, passed=len(checks))
    output=root/'verification/formulation_checks.json'
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=len(checks),output=str(output),scope=result['scope'])))


if __name__ == '__main__':
    main()
