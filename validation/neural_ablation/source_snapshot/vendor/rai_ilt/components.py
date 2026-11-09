"""Nonnegative variable projection for a finite, shared exponential mixture.

This is an established separable least-squares construction, not a claim of a
new algorithm or a global optimum. BIC is an order-selection heuristic here:
nonnegative mixtures have singular boundaries and are not regular BIC models.
"""
import numpy as np
from scipy.optimize import minimize, nnls


def exponential_kernel(b, rates):
    with np.errstate(over="ignore"):
        q = np.asarray(b)[:, None] * np.asarray(rates)[None, :]
    return np.exp(-q)


def _profile(z, b, y, sigma):
    """Eliminate amplitudes by weighted NNLS; differentiate the value function."""
    rates = np.exp(z)
    kernel = exponential_kernel(b, rates)
    amplitudes = np.column_stack([
        nnls(kernel / sigma[:, j, None], y[:, j] / sigma[:, j],
             maxiter=max(100, 10*len(z)))[0] for j in range(y.shape[1])
    ])
    residual = (kernel @ amplitudes-y) / sigma
    with np.errstate(over="ignore"):
        q = b[:, None]*rates
    derivative = -np.minimum(q, 745)*kernel
    gradient = np.sum((derivative.T @ (residual/sigma))*amplitudes, axis=1)
    return .5*float(np.sum(residual**2)), gradient, amplitudes


def fit_components(b, raw_y, raw_sigma, bounds, config):
    """Fit orders 0..K without access to true rates, amplitudes or component count.

All columns share rates but have independent nonnegative amplitudes, including
zero amplitudes. No graph or normalized-shape similarity is needed. All rows
are used for fitting and BIC; these scores are not held-out validation results.
    """
    # Dimensionless rate coordinates and column normalization preserve units,
    # amplitude scaling and the original weighted least-squares objective.
    reference = np.exp(np.mean(np.log(bounds)))
    t = b*reference
    lo, hi = np.log(np.asarray(bounds)/reference)
    scales = np.maximum(np.max(np.abs(raw_y), axis=0), np.median(raw_sigma, axis=0))
    y, sigma = raw_y/scales, raw_sigma/scales
    n, p = y.shape
    max_order = min(config.max_components, (n*p-1)//(p+1))
    null_rss = float(np.sum((y/sigma)**2))
    candidates = [{"order": 0, "rss": null_rss, "bic": null_rss,
                   "parameters": 0, "success": True, "iterations": 0,
                   "message": "Zero-signal model", "projected_gradient_inf": 0.,
                   "starts_converged": 1, "starts": 1}]
    fits = [(np.empty(0), np.empty((0, p)))]
    previous = None
    for k in range(1, max_order+1):
        # Seed is fixed and independent of observations/truth/column order.
        rng = np.random.default_rng(config.component_seed+k)
        starts = [np.linspace(lo, hi, k+2)[1:-1]]
        if previous is not None and k > 1:
            # Inherit the previous solution and propose residual-driven births.
            grid = np.linspace(lo, hi, 49)
            scored = [( _profile(np.r_[previous, z], t, y, sigma)[0], z)
                      for z in grid]
            for _, z in sorted(scored)[:min(3, config.component_starts-1)]:
                starts.append(np.sort(np.r_[previous, z]))
        while len(starts) < config.component_starts:
            starts.append(np.sort(rng.uniform(lo, hi, k)))
        opts = []
        for initial in starts:
            def fun(z):
                value, gradient, _ = _profile(z, t, y, sigma)
                return value, gradient
            opt = minimize(fun, initial, jac=True, method="L-BFGS-B",
                           bounds=[(lo, hi)]*k,
                           options={"maxiter": config.max_iterations,
                                    "ftol": config.ftol, "gtol": config.gtol,
                                    "maxls": 40})
            opts.append(opt)
        opt = min(opts, key=lambda item: item.fun)
        lowest_value = float(opt.fun)
        # Roundoff can make an abnormal line-search stop microscopically lower
        # than an independently converged start. Prefer the converged estimate
        # only within the requested objective tolerance; never relabel a failure.
        near = [o for o in opts if o.success and
                o.fun <= lowest_value+10*config.ftol*max(1., abs(lowest_value))]
        if near:
            opt = min(near, key=lambda item: item.fun)
        value, gradient, amps = _profile(opt.x, t, y, sigma)
        permutation = np.argsort(opt.x)
        previous = opt.x[permutation]
        rates, amps = np.exp(previous)*reference, amps[permutation]*scales
        # Count every amplitude, including inactive ones. Conservative for sparse
        # DOSY spectra; avoids reducing the penalty opportunistically at zeros.
        parameters = k*(p+1)
        projected = opt.x-np.clip(opt.x-gradient, lo, hi)
        candidates.append({"order": k, "rss": 2*value,
                           "bic": 2*value+parameters*np.log(n*p),
                           "parameters": parameters, "success": bool(opt.success),
                           "message": str(opt.message), "iterations": int(opt.nit),
                           "projected_gradient_inf": float(np.max(np.abs(projected))),
                           "starts_converged": sum(bool(o.success) for o in opts),
                           "starts": len(opts), "rates": rates.tolist()})
        candidates[-1]["converged_start_objective_difference"] = float(value-lowest_value)
        fits.append((rates, amps))
    selected = int(np.argmin([item["bic"] for item in candidates]))
    best = candidates[selected]
    for item in candidates:
        item["delta_bic"] = float(item["bic"]-best["bic"])
        item["accepted"] = item["order"] == selected
    rates, amps = fits[selected]
    rss = best["rss"]
    dof = n*p-best["parameters"]
    # Local rate information after projecting out amplitudes. This is a
    # conditioning diagnostic, not a calibrated confidence interval, especially
    # when NNLS is on a boundary or model order is uncertain.
    information = np.zeros((selected, selected))
    if selected:
        kernel = exponential_kernel(b, rates)
        q = np.minimum(b[:, None]*rates, 745)
        derivative = -q*kernel
        for j in range(p):
            a = kernel/raw_sigma[:, j, None]
            active = amps[:, j] > 0
            g = derivative*amps[:, j]/raw_sigma[:, j, None]
            nuisance = a[:, active]
            if nuisance.shape[1]:
                g -= nuisance @ np.linalg.lstsq(nuisance, g, rcond=None)[0]
            information += g.T@g
        eigen = np.linalg.eigvalsh(information)
        condition = float(eigen[-1]/eigen[0]) if eigen[0] > 0 else None
        local_sd = (np.sqrt(np.maximum(0, np.diag(np.linalg.inv(information)))).tolist()
                    if eigen[0] > 1e-12*eigen[-1] else None)
    else:
        condition, local_sd = None, []
    diagnostics = {
        "selected_components": selected,
        "selection_criterion": "known-noise BIC heuristic: RSS + k*(signals+1)*log(observations)",
        "selection_uses_all_rows": True,
        "model_order_ambiguous": sum(c["delta_bic"] <= 2 for c in candidates) > 1,
        "model_order_search_complete": all(c["success"] for c in candidates),
        "component_limit_reached": selected == max_order,
        "reduced_chi_square": rss/dof,
        "noise_fit_adequate": bool(rss <= dof+3*np.sqrt(2*dof)),
        "rate_information_condition": condition,
        "local_log_rate_sd": local_sd,
        "rate_bound_hit": bool(selected and np.any(
            (rates <= bounds[0]*(1+1e-5)) | (rates >= bounds[1]*(1-1e-5)))),
        "local_rate_precision_poor": bool(selected and
            (local_sd is None or max(local_sd) > .15)),
        "uncertainty_interpretation": "local curvature only; not confidence intervals or a separation certificate",
        "scales": scales.tolist(),
        "optimizer": {key: best[key] for key in
                      ("success", "message", "iterations", "projected_gradient_inf")},
    }
    diagnostics["optimizer"]["objective"] = .5*rss
    diagnostics["optimizer"]["algorithm"] = "multistart nonnegative variable projection"
    return rates, amps, candidates, diagnostics
