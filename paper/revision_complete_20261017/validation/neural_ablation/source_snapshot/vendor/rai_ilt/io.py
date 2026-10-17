"""Explicit canonical MATLAB interchange; input files are never overwritten."""
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
from .model import DecayProblem


def load_problem(path):
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(128)
    if header[:8] == b"\x89HDF\r\n\x1a\n" or header.startswith(b"MATLAB 7.3 MAT-file"):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("MATLAB v7.3 requires optional h5py; install rai-ilt[mat73]") from exc
        with h5py.File(path, "r") as f:
            data = {}
            for key in ("Y", "b", "sigma", "rate_bounds", "ppm", "covariance"):
                if key in f:
                    if not isinstance(f[key], h5py.Dataset) or f[key].dtype.kind not in "fiu":
                        raise ValueError(f"Unsupported v7.3 variable {key}; use numeric canonical arrays")
                    data[key] = np.asarray(f[key]).T  # MATLAB reverses dimension ordering.
    else:
        data = loadmat(path, simplify_cells=True)
    missing = {"Y", "b", "sigma", "rate_bounds"} - data.keys()
    if "covariance" in data:
        raise ValueError("Full noise covariance is not supported; sigma denotes independent measurement errors")
    if missing:
        raise ValueError("Missing canonical variables: " + ", ".join(sorted(missing)))
    return DecayProblem(data["b"], data["Y"], data["sigma"], data["rate_bounds"],
                        ppm=data.get("ppm"), metadata={"source": str(path.resolve())})


def save_result(path, result, problem, *, overwrite=False):
    from . import __version__
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
    source = problem.metadata.get("source")
    if source and path.resolve() == Path(source).resolve():
        raise ValueError("An input file cannot be used as output")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"rate_edges": result.rate_edges, "rates": result.rates, "masses": result.masses,
                   "density_log_rate": result.density_log_rate, "reconstructed": result.reconstructed,
                   "residual": result.residual, "graph": result.graph, "b": problem.b, "sigma": problem.sigma,
                   "Y": problem.y, "rate_bounds": problem.rate_bounds,
                   "diagnostics_json": json.dumps(result.diagnostics), "history_json": json.dumps(result.history),
                   "config_json": json.dumps(result.config), "metadata_json": json.dumps(problem.metadata),
                   "method": result.diagnostics.get("method", "RAI-ILT"), "version": __version__,
                   "representation": result.config["representation"]}
    if result.component_rates is not None:
        data["component_rates"] = result.component_rates
        data["component_amplitudes"] = result.component_amplitudes
    if result.config["representation"] == "moments":
        data["moments"] = result.moments
        data["moment_region_bounds"] = np.array([[r['lower'],r['upper']] for r in result.moment_regions])
        data["moment_region_degrees"] = np.array([r['degree'] for r in result.moment_regions],dtype=int)
        data["quadrature_rates"] = result.quadrature_rates
        data["quadrature_amplitudes"] = result.quadrature_amplitudes
        data["component_candidate_rates"] = np.array(result.diagnostics['component_candidate_rates'])
        data["component_candidate_amplitudes"] = np.array(result.diagnostics['component_candidate_amplitudes'])
        if result.seed_moments is not None:
            data["seed_moments"] = result.seed_moments
            data["seed_quadrature_rates"] = result.seed_quadrature_rates
            data["seed_quadrature_amplitudes"] = result.seed_quadrature_amplitudes
    if result.config["representation"] == "adaptive":
        data["active_leaves_zero_based_half_open"] = np.asarray(result.diagnostics["active_leaves"], dtype=int)
        data["fine_bins"] = result.diagnostics["fine_bins"]
        data["active_bins"] = result.diagnostics["active_bins"]
    if problem.ppm is not None:
        data["ppm"] = problem.ppm
    savemat(path, data, do_compression=True)
