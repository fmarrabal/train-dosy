"""Positive quadratic subproblems with held-out acquisition refinement.

No novelty, statistical coverage, or continuous-identifiability guarantee is implied.
"""
from dataclasses import dataclass, asdict
import time
import numpy as np
from scipy.optimize import minimize
from .model import DecayProblem, bin_kernel, log_edges, regularizers, refine_bins


@dataclass(frozen=True)
class RAIConfig:
    representation: str = "components"
    max_components: int = 5
    component_starts: int = 10
    component_seed: int = 1729
    initial_bins: int = 128
    max_bins: int = 256
    max_levels: int = 5
    smoothness: float = 0.02
    ridge: float = 1e-5
    coupling: float = 0.05
    graph_threshold: float = 1.5
    graph_min_snr: float = 8.0
    graph_max_neighbors: int = 3
    validation_stride: int = 4
    min_validation_gain: float = 0.01
    min_split_information: float = 0.25
    max_iterations: int = 3000
    ftol: float = 1e-12
    gtol: float = 1e-6

    def __post_init__(self):
        if self.representation not in ("components", "density"):
            raise ValueError("representation must be 'components' or 'density'")
        for key in ("initial_bins", "max_bins", "max_levels", "graph_max_neighbors", "validation_stride", "max_iterations", "max_components", "component_starts", "component_seed"):
            value = getattr(self, key)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise ValueError(f"{key} must be an integer")
        if self.max_components < 1 or self.component_starts < 1 or self.component_seed < 0:
            raise ValueError("Invalid component search configuration")
        if self.initial_bins < 3 or self.max_bins < self.initial_bins or self.max_levels < 0:
            raise ValueError("Invalid bin or refinement limits")
        if self.validation_stride < 3 or self.max_iterations < 1 or self.graph_max_neighbors < 0:
            raise ValueError("Invalid iteration, validation, or graph limits")
        for key in ("smoothness", "ridge", "coupling", "graph_threshold", "graph_min_snr", "min_validation_gain", "min_split_information", "ftol", "gtol"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.ridge <= 0 or self.ftol <= 0 or self.gtol <= 0:
            raise ValueError("ridge, ftol and gtol must be positive")


@dataclass
class RAIResult:
    rate_edges: np.ndarray
    masses: np.ndarray
    reconstructed: np.ndarray
    residual: np.ndarray
    graph: np.ndarray
    history: list
    diagnostics: dict
    config: dict
    component_rates: np.ndarray | None = None
    component_amplitudes: np.ndarray | None = None

    def predict(self, b):
        """Evaluate the fitted model, retaining off-grid rates when available."""
        from .model import real_array
        from .components import exponential_kernel
        b = real_array(b, "b").reshape(-1)
        if np.any(b < 0):
            raise ValueError("b must be nonnegative")
        if self.component_rates is not None:
            return exponential_kernel(b, self.component_rates) @ self.component_amplitudes
        return bin_kernel(b, self.rate_edges) @ self.masses

    @property
    def rates(self):
        return np.sqrt(self.rate_edges[:-1] * self.rate_edges[1:])

    @property
    def density_log_rate(self):
        return self.masses / np.diff(np.log(self.rate_edges))[:, None]


def build_graph(y, sigma, config):
    """Compare scaled decay shapes. Similarity is not chemical identification.

    All information comes from the supplied rows (training rows during selection).
    Mutual neighbor selection keeps ties without using column order to break them.
    """
    p = y.shape[1]
    graph = np.zeros((p, p))
    if p == 1 or config.coupling == 0 or config.graph_max_neighbors == 0:
        return graph
    scale = np.maximum(np.max(np.abs(y), axis=0), np.finfo(float).tiny)
    snr = scale / np.median(sigma, axis=0)
    score = np.full((p, p), np.inf)
    for j in range(p):
        if snr[j] < config.graph_min_snr:
            continue
        for k in range(j):
            if snr[k] < config.graph_min_snr:
                continue
            # Compare shapes after symmetric amplitude normalization.
            vj, vk = y[:, j] / scale[j], y[:, k] / scale[k]
            sj, sk = sigma[:, j] / scale[j], sigma[:, k] / scale[k]
            # Propagate uncertainty of the scale observation conservatively.
            var = sj**2 + sk**2 + vj**2 * sj.max()**2 + vk**2 * sk.max()**2
            s = np.mean((vj - vk)**2 / var)
            if s <= config.graph_threshold:
                score[j, k] = score[k, j] = s
    selected = np.zeros((p, p), bool)
    for j in range(p):
        finite = np.sort(score[j, np.isfinite(score[j])])
        if len(finite):
            cutoff = finite[min(config.graph_max_neighbors, len(finite)) - 1]
            selected[j] = score[j] <= cutoff
    mutual = selected & selected.T
    graph[mutual] = np.exp(-score[mutual])
    return graph


def _objective(x, k, y, sigma, r, inv_width, laplacian, config):
    n = y.shape[0]
    residual = k @ x - y
    weighted = residual / sigma
    rx = r @ x
    value = .5 * np.sum(weighted**2) / n + .5 * config.smoothness * np.sum(rx**2)
    value += .5 * config.ridge * np.sum(inv_width[:, None] * x**2)
    value += .5 * config.coupling * np.sum(inv_width[:, None] * x * (x @ laplacian))
    grad = k.T @ (residual / sigma**2) / n + config.smoothness * (r.T @ rx)
    grad += config.ridge * inv_width[:, None] * x
    grad += config.coupling * inv_width[:, None] * (x @ laplacian)
    return float(value), grad


def _fit(b, y, sigma, edges, graph, config, initial=None):
    k = bin_kernel(b, edges)
    r, inv_width = regularizers(edges)
    laplacian = np.diag(graph.sum(axis=1)) - graph
    shape = (len(edges) - 1, y.shape[1])
    x0 = np.full(shape, 1 / shape[0]) if initial is None else np.maximum(initial, 0)
    # Exact Hessian diagonal, used only as a change of optimization coordinates.
    # This improves conditioning on fine grids without changing the objective.
    diagonal = (k*k).T @ (1/sigma**2) / len(b)
    diagonal += config.smoothness*np.sum(r*r, axis=0)[:, None]
    diagonal += config.ridge*inv_width[:, None]
    diagonal += config.coupling*inv_width[:, None]*np.diag(laplacian)[None, :]
    scale = np.sqrt(diagonal)
    def fun(flat):
        value, grad = _objective(flat.reshape(shape)/scale, k, y, sigma, r, inv_width, laplacian, config)
        return value, (grad/scale).ravel()
    opt = minimize(fun, (x0*scale).ravel(), jac=True, method="L-BFGS-B", bounds=[(0, None)] * x0.size,
                   options={"maxiter": config.max_iterations, "ftol": config.ftol, "gtol": config.gtol, "maxls": 40})
    x = opt.x.reshape(shape)/scale
    _, grad = _objective(x, k, y, sigma, r, inv_width, laplacian, config)
    projected = x-np.maximum(0, x-grad)
    return x, {"success": bool(opt.success), "message": str(opt.message), "iterations": int(opt.nit),
               "objective": float(opt.fun), "projected_gradient_inf": float(np.max(np.abs(projected))),
               "preconditioner": "exact Hessian diagonal coordinate scaling"}


def _split_candidates(b, y, sigma, edges, x, config):
    """Residual contrast scores, screened by noise-scaled distinguishability."""
    z = log_edges(edges)
    all_edges, _ = refine_bins(edges, x, range(len(x)))
    child_kernel = bin_kernel(b, all_edges)
    contrast = (child_kernel[:, 0::2] - child_kernel[:, 1::2]) / 2
    residual = y - bin_kernel(b, edges) @ x
    score = np.zeros(len(x))
    for p in range(y.shape[1]):
        a = contrast / sigma[:, p, None]
        energy = np.sum(a*a, axis=0)
        corr = a.T @ (residual[:, p] / sigma[:, p])
        score += corr**2 / np.maximum(energy, 1e-30)
    info = np.max(np.sqrt(np.mean((contrast[:, :, None] * x[None, :, :] / sigma[:, None, :])**2, axis=0)), axis=1)
    eligible = (info >= config.min_split_information) & (np.diff(z) > 1e-5)
    candidates = np.flatnonzero(eligible)
    budget = min(max(1, len(x)//3), config.max_bins - len(x))
    return candidates[np.argsort(-score[candidates], kind="stable")[:budget]].tolist()


def solve(problem: DecayProblem, config: RAIConfig | None = None) -> RAIResult:
    """Fit a finite mixture (default) or an explicitly requested smooth density.

    Components share continuous rates across DOSY columns. The component count
    is selected from the data; the default upper search limit is five.
    """
    from .adaptive import AdaptiveConfig, solve_adaptive
    from .moment_solver import MomentConfig
    from .moments import solve_moments
    if isinstance(config, MomentConfig):
        return solve_moments(problem, config)
    if isinstance(config, AdaptiveConfig):
        return solve_adaptive(problem, config)
    config = config or RAIConfig()
    if not isinstance(problem, DecayProblem):
        raise TypeError("problem must be a DecayProblem")
    if config.representation == "density":
        return _solve_density(problem, config)
    from .components import fit_components, exponential_kernel
    start = time.perf_counter()
    order = np.argsort(problem.b, kind="stable")
    rates, amplitudes, history, diagnostics = fit_components(
        problem.b[order], problem.y[order], problem.sigma[order], problem.rate_bounds, config)
    edges = np.geomspace(*problem.rate_bounds, config.initial_bins+1)
    masses = np.zeros((config.initial_bins, problem.y.shape[1]))
    # Atomic masses are assigned to containing bins for export/display only.
    # The exact forward prediction always uses the continuous rates.
    indices = np.clip(np.searchsorted(edges, rates, side="right")-1, 0, config.initial_bins-1)
    np.add.at(masses, indices, amplitudes)
    reconstruction = exponential_kernel(problem.b, rates) @ amplitudes
    residual = problem.y-reconstruction
    diagnostics.update({"representation": "components",
        "mode": "single" if problem.y.shape[1] == 1 else "dosy",
        "rate_grid_role": "mass histogram only; optimization and prediction are off-grid",
        "refinement_stop": "continuous_rate_optimization", "graph_edges": 0,
        "seconds": time.perf_counter()-start,
        "weighted_rmse": float(np.sqrt(np.mean((residual/problem.sigma)**2))),
        "selection_rows_original": [], "statistical_coverage_claimed": False,
        "continuous_identifiability_certified": False,
        "noise_model": "known diagonal standard deviations"})
    return RAIResult(edges, masses, reconstruction, residual,
                     np.zeros((problem.y.shape[1], problem.y.shape[1])),
                     history, diagnostics, asdict(config), rates, amplitudes)


def _solve_density(problem, config):
    start = time.perf_counter()
    order = np.argsort(problem.b, kind="stable")
    b, raw_y, raw_s = problem.b[order], problem.y[order], problem.sigma[order]
    n = len(b)
    validation = np.zeros(n, bool)
    if n >= 12 and config.max_levels:
        validation[2:-1:config.validation_stride] = True
    train = ~validation
    # During selection the scale and graph use training observations only.
    scale = np.maximum(np.max(np.abs(raw_y[train]), axis=0), np.median(raw_s[train], axis=0))
    y, sigma = raw_y / scale, raw_s / scale
    edges = np.geomspace(*problem.rate_bounds, config.initial_bins + 1)
    graph = build_graph(y[train], sigma[train], config)
    x, diag = _fit(b[train], y[train], sigma[train], edges, graph, config)
    def validation_loss(e, xx):
        return float(np.mean(((bin_kernel(b[validation], e) @ xx - y[validation]) / sigma[validation])**2))
    loss = validation_loss(edges, x) if validation.any() else None
    history = [{"bins": len(x), "accepted": True, "validation_loss": loss, "optimizer": diag}]
    reason = "no_validation_split" if not validation.any() else "max_levels"
    for level in range(config.max_levels if validation.any() else 0):
        if len(x) >= config.max_bins:
            reason = "max_bins"; break
        selected = _split_candidates(b[train], y[train], sigma[train], edges, x, config)
        if not selected:
            reason = "no_informative_split"; break
        new_edges, warm = refine_bins(edges, x, selected)
        new_x, new_diag = _fit(b[train], y[train], sigma[train], new_edges, graph, config, warm)
        new_loss = validation_loss(new_edges, new_x)
        accepted = new_diag["success"] and new_loss < loss * (1 - config.min_validation_gain)
        history.append({"bins": len(new_x), "split_bins": selected, "accepted": bool(accepted),
                        "validation_loss": new_loss, "optimizer": new_diag})
        if not accepted:
            reason = "validation_no_gain" if new_diag["success"] else "candidate_optimizer_failed"
            break
        edges, x, loss = new_edges, new_x, new_loss
    # Refit selected resolution using all data. Validation score above is a selection
    # diagnostic, not an independent estimate of generalization after this refit.
    full_graph = build_graph(y, sigma, config)
    x, final_diag = _fit(b, y, sigma, edges, full_graph, config, x)
    masses = x * scale
    reconstruction = bin_kernel(problem.b, edges) @ masses
    residual = problem.y - reconstruction
    diagnostics = {"representation": "density", "mode": "single" if y.shape[1] == 1 else "dosy", "refinement_stop": reason,
                   "optimizer": final_diag, "seconds": time.perf_counter() - start,
                   "weighted_rmse": float(np.sqrt(np.mean((residual / problem.sigma)**2))),
                   "selection_rows_original": order[validation].tolist(),
                   "graph_edges": int(np.count_nonzero(np.triu(full_graph, 1))),
                   "scales": scale.tolist(), "statistical_coverage_claimed": False,
                   "continuous_identifiability_certified": False, "noise_model": "known diagonal standard deviations"}
    return RAIResult(edges, masses, reconstruction, residual, full_graph, history, diagnostics, asdict(config))
