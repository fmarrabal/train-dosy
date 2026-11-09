"""RAI-S / DOME-S: validated spectral support for shared discrete DOSY rates.

No truth, outer test data, or relative-to-tallest-peak cutoff is accepted.
The conditional Gaussian score is a screening heuristic with estimated rates;
it is NOT a calibrated chemical detection confidence level. Reserved gradients
choose among screened and unscreened candidates before a final NNLS refit.
"""
from pathlib import Path
import os, sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent/'vendor'))
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm
from dome.solver import amplitudes_exact


@dataclass(frozen=True)
class Config:
    max_components: int = 4
    alpha: float = .01
    validation_stride: int = 4
    selection_se: float = 1.
    max_nfev: int = 350


def groups_from_ppm(ppm):
    """Contiguous acquired signal windows; never bridge solvent/noise gaps."""
    ppm = np.asarray(ppm)
    if len(ppm) < 2:
        return np.zeros(len(ppm), int)
    order = np.argsort(ppm)
    steps = np.diff(ppm[order])
    if np.any(steps <= 0):
        raise ValueError('ppm coordinates must be distinct')
    step = np.percentile(steps, 25)
    g = np.r_[0, np.cumsum(steps > 1.6*step)]
    result = np.empty(len(ppm), int); result[order] = g
    return result


def nnls_support(k, y, support):
    a = np.zeros((k.shape[1], y.shape[1]))
    if not k.shape[1]:
        return a
    patterns, inverse = np.unique(support.T, axis=0, return_inverse=True)
    for i, pattern in enumerate(patterns):
        idx = np.flatnonzero(pattern); cols = np.flatnonzero(inverse == i)
        if len(idx):
            a[np.ix_(idx, cols)] = amplitudes_exact(k[:, idx], y[:, cols])
    return a


def profile(z, b, y, sigma, support, jacobian=False):
    d = np.exp(z); q = b[:, None]*d
    k = np.exp(-q); dk = -q*k
    a = nnls_support(k, y, support)
    r = (k@a-y)/sigma
    if not jacobian:
        return r, a
    n, p = y.shape
    j = np.zeros((n, p, len(d)))
    active = a > 1e-12*max(float(a.max(initial=0)), 1e-300)
    patterns, inverse = np.unique(active.T, axis=0, return_inverse=True)
    for mask, pattern in enumerate(patterns):
        idx = np.flatnonzero(pattern); cols = np.flatnonzero(inverse == mask)
        if not len(idx):
            continue
        ks = k[:, idx]; pinv = np.linalg.pinv(ks, rcond=1e-12)
        for local, atom in enumerate(idx):
            v = dk[:, atom]-ks@(pinv@dk[:, atom])
            j[:, cols, atom] = v[:, None]*a[atom, cols]/sigma - pinv.T[:, [local]]*(dk[:, atom]@r[:, cols])[None, :]
    return r, a, j.reshape(n*p, len(d))


def refine(b, y, sigma, bounds, d, support, max_nfev=350):
    live = support.any(axis=1)
    d = np.asarray(d)[live]; support = support[live]
    if not len(d):
        return dict(D=d, A=np.zeros((0, y.shape[1])), support=support,
                    prediction=np.zeros_like(y), success=True, optimality=0., kkt=0., parameters=0)
    cache = {}
    def both(z):
        if 'z' not in cache or not np.array_equal(cache['z'], z):
            r, a, j = profile(z, b, y, sigma, support, True)
            cache.update(z=z.copy(), r=r, a=a, j=j)
        return cache
    fit = least_squares(lambda z: both(z)['r'].ravel(), np.log(d),
        jac=lambda z: both(z)['j'], bounds=np.log(bounds), method='trf',
        max_nfev=max_nfev, ftol=1e-11, xtol=1e-11, gtol=1e-8)
    r, a, j = profile(fit.x, b, y, sigma, support, True)
    k = np.exp(-b[:, None]*np.exp(fit.x))
    gradient_a = k.T@r/sigma
    scale = max(float(np.max(np.abs(k.T@y/sigma**2))), 1.)
    residual_kkt = np.where(a > 1e-10*max(a.max(), 1e-300), abs(gradient_a), np.maximum(-gradient_a, 0))
    residual_kkt[~support] = 0
    grad = j.T@r.ravel()
    pg = fit.x-np.clip(fit.x-grad, *np.log(bounds))
    # A scale-aware first-order residual supplements the library stop status.
    denom = max(1., np.linalg.norm(j, ord=2)*np.linalg.norm(r))
    stationarity = float(np.max(abs(pg))/denom)
    order = np.argsort(fit.x)
    return dict(D=np.exp(fit.x[order]), A=a[order], support=support[order],
        prediction=k@a, success=bool(fit.success and stationarity <= 1e-6),
        stop_success=bool(fit.success), stationarity=stationarity,
        optimality=float(np.max(abs(pg))), kkt=float(residual_kkt.max(initial=0)/scale),
        parameters=int(np.sum(support)+len(d)), nfev=int(fit.nfev))


def screen_support(b, y, sigma, d, groups, alpha):
    k = np.exp(-b[:, None]*np.asarray(d)); r = len(d)
    unique = np.unique(groups); scores = np.full((r, len(unique)), -np.inf)
    support = np.zeros((r, y.shape[1]), bool)
    threshold = float(norm.isf(alpha/max(1, r*len(unique))))
    for atom in range(r):
        nuisance = np.delete(k, atom, axis=1)
        v = k[:, atom].copy()
        if nuisance.shape[1]:
            v -= nuisance@np.linalg.lstsq(nuisance, v, rcond=1e-12)[0]
        vn = np.linalg.norm(v)
        if vn < 1e-10*np.linalg.norm(k[:, atom]):
            # Numerical non-identifiability is not evidence of absence.
            support[atom] = True
            continue
        z = v@y/(sigma*vn)
        for gi, group in enumerate(unique):
            cols = groups == group
            # Positive aggregate evidence retains small but coherent signals.
            scores[atom, gi] = z[cols].sum()/np.sqrt(cols.sum())
            support[atom, cols] = scores[atom, gi] > threshold
    return support, scores, threshold


def native_path(b, y, sigma, bounds, method, max_components):
    if method == 'RAI':
        from rai_ilt.components import fit_components
        from rai_ilt.solver import RAIConfig
        ss = np.full_like(y, sigma)
        d, a, candidates, diag = fit_components(b, y, ss, bounds, RAIConfig(max_components=max_components))
        path = [np.asarray(c.get('rates', [])) for c in candidates]
        return path, dict(D=d, A=a, diagnostics=diag)
    if method == 'DOME':
        from dome.solver import fit, Problem, Config as DomeConfig
        out = fit(Problem(b, y, sigma, bounds), DomeConfig(max_components=max_components))
        return [np.asarray(c['D']) for c in out['order_path']], dict(D=out['D'], A=out['amplitudes'], diagnostics={k:out[k] for k in ('selected_rank','optimizer_success','optimality')})
    raise ValueError('method must be RAI or DOME')


def fit_support(b, y, sigma, ppm, bounds=(.1e-9, 15e-9), *, method='RAI', config=None, extra_seeds=None):
    """Fit from observed acquisition rows only; b in s/m2, D in m2/s."""
    cfg = config or Config()
    b = np.asarray(b, float); y = np.asarray(y, float); ppm = np.asarray(ppm, float)
    if y.ndim != 2 or y.shape != (len(b), len(ppm)) or len(b) < 12:
        raise ValueError('Require at least 12 acquisitions and matching Y/ppm')
    if not np.isfinite(y).all() or not np.isfinite(b).all() or sigma <= 0 or not np.isfinite(sigma):
        raise ValueError('Finite observations and positive noise required')
    if np.any(b < 0) or not 0 < bounds[0] < bounds[1] or not 0 < cfg.alpha < 1:
        raise ValueError('Invalid b, bounds, or alpha')
    order = np.argsort(b); b = b[order]; y = y[order]
    if np.any(np.diff(b) <= 0):
        raise ValueError('Distinct b rows required')
    val = np.zeros(len(b), bool); val[2:-1:cfg.validation_stride] = True
    tr = ~val; groups = groups_from_ppm(ppm)
    path, native = native_path(b[tr], y[tr], sigma, bounds, method, cfg.max_components)
    if extra_seeds is not None:
        path += [np.asarray(s) for s in extra_seeds]
    candidates = []
    for d in path:
        raw = np.ones((len(d), y.shape[1]), bool)
        supports = [('unrestricted', raw, None, None)]
        if len(d):
            kept, scores, threshold = screen_support(b[tr], y[tr], sigma, d, groups, cfg.alpha)
            if not np.array_equal(kept, raw):
                supports += [('screened', kept, scores, threshold)]
        for kind, sup, scores, threshold in supports:
            if any(np.array_equal(c['initial_support'], sup) and np.array_equal(c['initial_D'],d) for c in candidates):
                continue
            q = refine(b[tr], y[tr], sigma, bounds, d, sup, cfg.max_nfev)
            pred = np.exp(-b[val, None]*q['D'])@q['A']
            q.update(kind=kind, val_prediction=pred, val_sse=float(np.sum(((pred-y[val])/sigma)**2)),
                     initial_D=d, initial_support=sup, group_scores=scores, threshold=threshold)
            candidates.append(q)
    good = [i for i,c in enumerate(candidates) if c['success'] and c['kkt'] <= 1e-7]
    if not good:
        raise RuntimeError('No verified candidate')
    best = min(good, key=lambda i:candidates[i]['val_sse'])
    eligible = []
    for i in good:
        c = candidates[i]
        delta = c['val_prediction']-candidates[best]['val_prediction']
        # Exact conditional noise SD of paired SSE difference, given fixed fits.
        se = 2*np.linalg.norm(delta)/sigma
        c['paired_se'] = float(se)
        if c['val_sse'] <= candidates[best]['val_sse']+cfg.selection_se*se+1e-10:
            eligible.append(i)
    # First prefer fewer distinct diffusion rates, then a smaller spectral
    # support. A new weak atom must not win merely by replacing a few pixels.
    selected = min(eligible, key=lambda i:(len(candidates[i]['D']),candidates[i]['parameters'],candidates[i]['val_sse']))
    chosen = candidates[selected]
    result = refine(b,y,sigma,bounds,chosen['D'],chosen['support'],cfg.max_nfev)
    rss = float(np.sum(((result['prediction']-y)/sigma)**2))
    dof = max(y.size-result['parameters'],1)
    result['reduced_chi_square'] = rss/dof
    result['atomic_model_mismatch_flag'] = bool(rss > dof+3*np.sqrt(2*dof))
    result['bound_hit'] = bool(np.any((result['D'] <= bounds[0]*1.001)|(result['D'] >= bounds[1]/1.001)))
    if len(result['D']):
        _,_,jac = profile(np.log(result['D']),b,y,sigma,result['support'],True)
        singular = np.linalg.svd(jac,compute_uv=False)
        result['rate_information_condition'] = float((singular[0]/max(singular[-1],1e-300))**2)
        result['local_log_rate_sd'] = np.sqrt(np.diag(np.linalg.pinv(jac.T@jac,rcond=1e-12)))
        result['poor_rate_precision'] = bool(result['rate_information_condition']>1e10 or np.max(result['local_log_rate_sd'])>.25)
    else:
        result['rate_information_condition'] = None;result['local_log_rate_sd']=[];result['poor_rate_precision']=False
    result['prediction'] = result['prediction'][np.argsort(order)]
    result.update(method=method+'-S', config=asdict(cfg), groups=groups,
                  selection=selected, candidate_kind=chosen['kind'], validation_best=best,
                  candidate_diagnostics=[{k:c[k] for k in ('D','kind','parameters','val_sse','success','kkt','group_scores','threshold')} for c in candidates],
                  inner_validation_rows=order[val], train_only_native=native,
                  warning='Conditional support scores use estimated rates; no formal detection guarantee.')
    return result
