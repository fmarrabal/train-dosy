"""Physical operator with integrated bin masses, piecewise uniform in log rate."""
from dataclasses import dataclass, field
import numpy as np
from numpy.polynomial.legendre import leggauss


def real_array(value, name):
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real; phase-correct data before inversion")
    try:
        out = np.array(arr, dtype=float, copy=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    return out


@dataclass(frozen=True)
class DecayProblem:
    b: np.ndarray
    y: np.ndarray
    sigma: np.ndarray | float
    rate_bounds: tuple[float, float]
    ppm: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        b = real_array(self.b, "b").reshape(-1)
        y = real_array(self.y, "Y")
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2 or y.shape[0] != len(b) or min(y.shape) < 1:
            raise ValueError("Y must have shape (number of acquisitions, number of signals)")
        if len(b) < 3 or np.any(b < 0) or len(np.unique(b)) < 3:
            raise ValueError("Need at least three distinct nonnegative acquisition values")
        bounds = real_array(self.rate_bounds, "rate_bounds").reshape(-1)
        if bounds.shape != (2,) or not 0 < bounds[0] < bounds[1]:
            raise ValueError("rate_bounds must be two ordered positive values")
        sigma = real_array(self.sigma, "sigma")
        if sigma.ndim == 1:
            if len(sigma) != len(b):
                raise ValueError("A 1D sigma is acquisition-wise; per-signal sigma must be (1,p)")
            sigma = sigma[:, None]
        try:
            sigma = np.broadcast_to(sigma, y.shape).copy()
        except ValueError as exc:
            raise ValueError("sigma must be scalar, acquisition-wise, or broadcastable to Y") from exc
        if np.any(sigma <= 0):
            raise ValueError("All noise standard deviations must be positive")
        ppm = None if self.ppm is None else real_array(self.ppm, "ppm").reshape(-1)
        if ppm is not None and len(ppm) != y.shape[1]:
            raise ValueError("ppm length must equal number of signals")
        for arr in (b, y, sigma, ppm):
            if arr is not None:
                arr.setflags(write=False)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "rate_bounds", tuple(bounds))
        object.__setattr__(self, "ppm", ppm)
        object.__setattr__(self, "metadata", dict(self.metadata))


def log_edges(rate_edges):
    edges = real_array(rate_edges, "rate_edges").reshape(-1)
    if len(edges) < 3 or np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
        raise ValueError("Need at least two bins with increasing positive edges")
    return np.log(edges)


def bin_kernel(b, rate_edges, quadrature_order=12):
    """Average exp(-b*r) over each log-rate bin; multiply by bin *mass*."""
    b = real_array(b, "b").reshape(-1)
    if np.any(b < 0):
        raise ValueError("Acquisitions must be nonnegative")
    if not isinstance(quadrature_order, int) or quadrature_order < 2:
        raise ValueError("quadrature_order must be an integer >=2")
    z = log_edges(rate_edges)
    q, w = leggauss(quadrature_order)
    nodes = (z[:-1, None] + z[1:, None]) / 2 + np.diff(z)[:, None] * q / 2
    return np.einsum("ijq,q->ij", np.exp(-b[:, None, None] * np.exp(nodes)[None, :, :]), w / 2)


def regularizers(rate_edges):
    """Finite-volume H1 seminorm and L2 norm of density per unit log rate."""
    z = log_edges(rate_edges)
    widths = np.diff(z)
    centers = (z[1:] + z[:-1]) / 2
    r = np.zeros((len(widths) - 1, len(widths)))
    idx = np.arange(len(widths) - 1)
    distances = np.sqrt(np.diff(centers))
    r[idx, idx] = -1 / widths[:-1] / distances
    r[idx, idx + 1] = 1 / widths[1:] / distances
    return r, 1 / widths


def refine_bins(rate_edges, masses, indices):
    """Split log bins at midpoints; preserve each column's total mass exactly."""
    z = log_edges(rate_edges)
    x = real_array(masses, "masses")
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] != len(z) - 1 or np.any(x < 0):
        raise ValueError("masses must be nonnegative with one row per bin")
    raw = list(indices)
    if any(not isinstance(i, (int, np.integer)) for i in raw):
        raise ValueError("Bin indices must be integers")
    selected = set(raw)
    if any(i < 0 or i >= len(z) - 1 for i in selected):
        raise ValueError("Bin index out of range")
    out_edges, out_x = [z[0]], []
    for j in range(len(z) - 1):
        if j in selected:
            out_edges.append((z[j] + z[j + 1]) / 2)
            out_x.extend([x[j] / 2, x[j] / 2])
        else:
            out_x.append(x[j])
        out_edges.append(z[j + 1])
    return np.exp(out_edges), np.asarray(out_x)
