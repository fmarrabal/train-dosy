"""DOME: experimental orthogonal moment-exchange inversion.

Original research design in this project, NOT a demonstrated novelty claim.
Variable projection, NNLS, trust regions, and moment-preserving split/merge
have prior art. The testable design is the paired curvature/spectral-contrast
split and moment-preserving exchange policy. Normal exp(-bD) acquisition only.
"""
from dataclasses import dataclass, asdict
from itertools import combinations
import time
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


@dataclass
class Problem:
    b: np.ndarray
    y: np.ndarray
    sigma: float
    bounds: tuple = (.06e-9, 5e-9)
    ppm: np.ndarray | None = None

    def __post_init__(self):
        self.b = np.asarray(self.b, float).ravel()
        self.y = np.asarray(self.y, float)
        if self.y.ndim == 1:
            self.y = self.y[:, None]
        if (self.y.ndim != 2 or self.y.shape[0] != len(self.b) or len(self.b) < 8 or
                self.y.shape[1] < 1 or not np.isfinite(self.y).all() or not np.isfinite(self.b).all()):
            raise ValueError('Require finite Y with shape (at least 8 b values, frequency columns)')
        if np.any(self.b < 0) or np.any(np.diff(self.b) <= 0):
            raise ValueError('b must be nonnegative and strictly increasing')
        if np.ndim(self.sigma) != 0 or not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError('sigma must be a positive scalar for independent equal-variance Gaussian noise')
        if len(self.bounds) != 2 or not np.isfinite(self.bounds).all() or not 0 < self.bounds[0] < self.bounds[1]:
            raise ValueError('Invalid positive diffusion bounds')
        self.bounds = tuple(float(x) for x in self.bounds)
        if self.ppm is not None:
            self.ppm = np.asarray(self.ppm, float).ravel()
            if self.ppm.size == 0:
                self.ppm = None
            elif len(self.ppm) != self.y.shape[1] or not np.isfinite(self.ppm).all():
                raise ValueError('ppm must match Y columns')
        self.sigma = float(self.sigma)
        self.scale = max(float(np.max(np.abs(self.y))), self.sigma)
        self.Y = self.y/self.scale
        self.noise = self.sigma/self.scale
        self.t = self.b*1e-9
        self.low, self.high = np.array(self.bounds)/1e-9
        self.L = float(np.log(self.high/self.low))

    def rates(self, u):
        return np.clip(self.low*np.exp(self.L*np.asarray(u)), self.low, self.high)

    def locations(self, d):
        return np.clip(np.log(np.asarray(d)/self.low)/self.L, 0, 1)

    def kernel(self, u):
        d = self.rates(u)
        q = self.t[:, None]*d
        k = np.exp(-q)
        return k, -q*self.L*k


@dataclass(frozen=True)
class Config:
    max_components: int = 4
    beam_width: int = 2
    refined_proposals: int = 5
    max_evaluations: int = 160
    complexity_weight: float = 2.
    moment_steps: bool = True
    exchanges: bool = True

    def __post_init__(self):
        for field in ('max_components', 'beam_width', 'refined_proposals', 'max_evaluations'):
            if type(getattr(self, field)) is not int or getattr(self, field) < 1:
                raise ValueError(field)
        if self.max_components > 6:
            raise ValueError('Exact active-subset solver currently supports at most six components')
        if not np.isfinite(self.complexity_weight) or self.complexity_weight <= 0:
            raise ValueError('complexity_weight')


def amplitudes_exact(k, y):
    """Solve small-q NNLS for all frequencies by enumerating active subsets.

    All 2**q subsets, including the empty one, are considered. This is a small
    component-count algorithm, not a 256-bin distribution solver. QR/SVD least
    squares avoids normal-equation conditioning when columns nearly coincide.
    """
    q, c = k.shape[1], y.shape[1]
    a = np.zeros((q, c)); best = np.sum(y*y, axis=0)
    for mask in range(1, 1 << q):
        idx = np.array([j for j in range(q) if mask & (1 << j)])
        ks = k[:, idx]
        candidate = np.linalg.lstsq(ks, y, rcond=1e-13)[0]
        valid = np.all(candidate >= -1e-12, axis=0)
        candidate = np.maximum(candidate, 0)
        residual = ks@candidate-y
        sse = np.sum(residual*residual, axis=0)
        good = valid & (sse < best)
        if np.any(good):
            cols = np.flatnonzero(good)
            a[:, cols] = 0
            a[np.ix_(idx, cols)] = candidate[:, cols]
            best[cols] = sse[cols]
    return a


def profile(p, u, jacobian=False):
    k, dk = p.kernel(u)
    a = amplitudes_exact(k, p.Y)
    r = k@a-p.Y
    if not jacobian:
        return float(np.sum(r*r)), a, r
    n, c = p.Y.shape; q = len(u)
    jac = np.zeros((n, c, q))
    active = a > 1e-12
    masks = np.sum(active*(1 << np.arange(q))[:, None], axis=0)
    for mask in np.unique(masks):
        if mask == 0:
            continue
        idx = np.array([j for j in range(q) if mask & (1 << j)])
        cols = np.flatnonzero(masks == mask)
        ks = k[:, idx]; inverse = np.linalg.pinv(ks, rcond=1e-13)
        for jlocal, j in enumerate(idx):
            projected = dk[:, j]-ks@(inverse@dk[:, j])
            jac[:, cols, j] = (projected[:, None]*a[j, cols][None, :] -
                              inverse.T[:, [jlocal]]*(dk[:, j]@r[:, cols])[None, :])
    return float(np.sum(r*r)), a, r, jac.reshape(n*c, q)


def refine(p, u, cfg):
    u = np.asarray(u, float)
    cache = {}
    def both(x):
        if 'x' not in cache or not np.array_equal(x, cache['x']):
            value, a, r, jac = profile(p, x, True)
            cache.update(x=x.copy(), value=value, a=a, r=r, jac=jac)
        return cache
    initial = profile(p, u)[0]
    result = least_squares(lambda x: both(x)['r'].ravel()/p.noise, u,
                           jac=lambda x: both(x)['jac']/p.noise, bounds=(0., 1.),
                           method='trf', max_nfev=cfg.max_evaluations, ftol=1e-10,
                           xtol=1e-10, gtol=1e-8)
    un = result.x
    if profile(p, un)[0] > initial + 1e-12:
        un = u
    un = np.sort(un)
    sse, a, r, jac = profile(p, un, True)
    singular = np.linalg.svd(jac/p.noise, compute_uv=False)
    projected_gradient = un-np.clip(un-(jac.T@r.ravel())/p.noise**2, 0, 1)
    return dict(u=un, D=p.rates(un)*1e-9, amplitudes=a*p.scale,
                normalized_amplitudes=a, prediction=(r+p.Y)*p.scale, sse=sse,
                singular_values=singular, optimality=float(np.max(np.abs(projected_gradient))),
                optimizer_success=bool(result.success), optimizer_message=str(result.message),
                evaluations=int(result.nfev))


def curvature_split_step(p, state, j):
    """A variance step is visible even when a coincident-pair separation has zero derivative."""
    k, _ = p.kernel(state['u']); a = state['normalized_amplitudes']
    r = k@a-p.Y
    # Project curvature away from existing amplitude and position directions.
    _, _, _, tangent = profile(p, state['u'], True)
    inverse = np.linalg.pinv(k, rcond=1e-13)
    curvature = .5*p.t**2*k[:, j]
    curvature -= k@(inverse@curvature)
    direction = (curvature[:, None]*a[[j], :]).ravel()
    direction -= tangent@(np.linalg.pinv(tangent, rcond=1e-11)@direction)
    denominator = float(direction@direction)
    variance = max(0., -float(r.ravel()@direction)/denominator) if denominator > 1e-25 else 0.
    # DOSY contrast can reveal separation at first order even if the weighted
    # sum of frequency gradients is zero at a fitted single shared rate.
    slope = -p.t*k[:, j]; slope -= k@(inverse@slope)
    contrast = slope@r
    z = -np.tanh(contrast/max(p.noise*np.linalg.norm(slope), 1e-15))
    direction_spectral = slope[:, None]*a[[j], :]*z[None, :]
    denominator2 = float(np.sum(direction_spectral**2))
    separation = max(0., -float(np.sum(r*direction_spectral))/denominator2) if denominator2 > 1e-25 else 0.
    return variance, separation, dict(curvature_score=float(-r.ravel()@direction),
                                      contrast_norm=float(np.linalg.norm(contrast)))


def split_proposals(p, state, cfg):
    d = p.rates(state['u']); candidates = []; diagnostics = []
    for j, mean in enumerate(d):
        if cfg.moment_steps:
            variance, spectral_delta, diag = curvature_split_step(p, state, j)
            diagnostics.append(dict(parent=j, predicted_variance=variance, predicted_spectral_half_separation=spectral_delta, **diag))
            scales = [np.sqrt(variance), spectral_delta]
            # Finite steps accompany the local moment expansion; exact forward
            # evaluation decides the ranking, never the Taylor model alone.
            scales += [mean*.12, mean*.4, mean*.85]
            weights = [.15, .5, .85]
        else:
            scales = [mean*.12, mean*.4, mean*.85]
            weights = [.5]
        for sd in scales:
            for fraction in weights:
                if not np.isfinite(sd) or sd <= 1e-10:
                    continue
                delta = sd/np.sqrt(fraction*(1-fraction))
                delta = min(delta, .95*(mean-p.low)/(1-fraction), .95*(p.high-mean)/fraction)
                if delta < 1e-6:
                    continue
                pair = [mean-(1-fraction)*delta, mean+fraction*delta]
                un = np.sort(p.locations(np.r_[np.delete(d, j), pair]))
                candidates.append(un)
    return candidates, diagnostics


def moment_exchange(p, state):
    """Keep aggregate mass, mean and variance while moving a scalar pair.

    In DOSY, candidate endpoints are shared but amplitudes are profiled anew.
    Aggregate moment identities refer to the *proposal*, not a constraint on
    the subsequent fitted solution or every individual spectral column.
    """
    d = p.rates(state['u']); mass = state['normalized_amplitudes'].sum(axis=1)
    candidates = []
    for i, j in combinations(range(len(d)), 2):
        total = mass[i]+mass[j]
        if total < 1e-12:
            continue
        mean = (mass[i]*d[i]+mass[j]*d[j])/total
        variance = (mass[i]*(d[i]-mean)**2+mass[j]*(d[j]-mean)**2)/total
        for fraction in [.08, .2, .5, .8, .92]:
            left = mean-np.sqrt(variance*(1-fraction)/fraction)
            right = mean+np.sqrt(variance*fraction/(1-fraction))
            if p.low < left < right < p.high:
                keep = np.ones(len(d), bool); keep[[i, j]] = False
                candidates.append(np.sort(p.locations(np.r_[d[keep], left, right])))
    return candidates


def distinct(states, count):
    kept = []
    for state in sorted(states, key=lambda x: x['sse']):
        if not any(np.max(np.abs(state['u']-old['u'])) < 1e-4 for old in kept):
            kept.append(state)
        if len(kept) == count:
            break
    return kept


def fit(problem, config=None):
    p = problem; cfg = config or Config(); started = time.perf_counter()
    # Only a monocomponent starting solution. No known rank, true D, native
    # answer, neural training set or reference solver enters the algorithm.
    scalar = minimize_scalar(lambda u: profile(p, np.array([u]))[0], bounds=(0, 1), method='bounded',
                             options={'xatol': 1e-10})
    first = refine(p, [scalar.x], cfg)
    null = dict(u=np.empty(0), D=np.empty(0), amplitudes=np.empty((0, p.y.shape[1])),
                normalized_amplitudes=np.empty((0, p.y.shape[1])), prediction=np.zeros_like(p.y),
                sse=float(np.sum(p.Y**2)), singular_values=np.empty(0), optimality=0.,
                optimizer_success=True, optimizer_message='Null model, exact evaluation', evaluations=1)
    beam = [first]; path = [null, first]; step_log = []
    for rank in range(2, cfg.max_components+1):
        proposals = []; moments = []
        for state in beam:
            candidates, diag = split_proposals(p, state, cfg)
            proposals.extend(candidates); moments.extend(diag)
        if not proposals:
            break
        scored = sorted([(profile(p, u)[0], u) for u in proposals], key=lambda item: item[0])
        seeds = []
        for _, u in scored:
            if not any(np.max(np.abs(u-old)) < .01 for old in seeds):
                seeds.append(u)
            if len(seeds) >= cfg.refined_proposals:
                break
        refined = [refine(p, u, cfg) for u in seeds]
        beam = distinct(refined, cfg.beam_width)
        before = beam[0]['sse']; attempts = 0
        if cfg.exchanges:
            exchanges = moment_exchange(p, beam[0])
            best_seeds = sorted(exchanges, key=lambda u: profile(p, u)[0])[:cfg.refined_proposals]
            attempts = len(best_seeds)
            beam = distinct(beam+[refine(p, u, cfg) for u in best_seeds], cfg.beam_width)
        path.append(beam[0])
        step_log.append(dict(rank=rank, split_proposals=len(proposals), refined_splits=len(seeds),
                             exchange_attempts=attempts, sse_before_exchange=before,
                             sse_after_exchange=beam[0]['sse'], moment_directions=moments))
    # Approximate Gaussian information criterion with known sigma and active
    # amplitudes. Not an exact nonlinear SURE theorem or confidence level.
    # This conventional selector is exposed so the new step can be ablated.
    N = p.Y.size
    for state in path:
        q = len(state['u'])
        df = int(np.sum(state['normalized_amplitudes'] > 1e-10))+q
        penalty = cfg.complexity_weight*df + (2*df*(df+1)/max(N-df-1, 1))
        state['effective_parameters_approx'] = df
        state['selection_score'] = state['sse']/p.noise**2 + penalty
    selected = int(np.argmin([s['selection_score'] for s in path]))
    result = path[selected].copy()
    # Rank alternatives are reported without silently averaging different
    # chemical explanations into a smooth artificial distribution.
    minimum = result['selection_score']
    result.update(selected_rank=len(result['D']), order_path=[dict(rank=len(s['D']),
                  score=s['selection_score'], delta_score=s['selection_score']-minimum,
                  D=s['D'], amplitudes=s['amplitudes'], sse=s['sse'],
                  effective_parameters_approx=s['effective_parameters_approx'],
                  optimizer_success=s['optimizer_success'], optimality=s['optimality']) for s in path],
                  step_log=step_log, config=asdict(cfg), seconds=time.perf_counter()-started,
                  mode='single' if p.y.shape[1] == 1 else 'joint_dosy',
                  novelty_status='Research hypothesis; prior art on constituent methods acknowledged; novelty unproven')
    return result


def torch_forward(b, D, amplitudes):
    import torch
    return torch.exp(-b.reshape(-1, 1)*D.reshape(1, -1))@amplitudes
